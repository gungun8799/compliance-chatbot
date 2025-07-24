async def handle_standard_query(message: cl.Message):
    """Handles a standard, non-clarification query with hierarchical clarification."""
    import re
    from collections import defaultdict
    from difflib import SequenceMatcher
    import statistics
    # At the top of handle_standard_query
    current_q = message.content.strip()
    prev_q    = cl.user_session.get("pre_drill_query")
    
     # ─── 11) Fuzzy-fallback & final LLM answer ─────────────────────────────────
    # compute how close we are to any of your canned Q→A
    fuzzy_scores = {
        q: SequenceMatcher(None, current_q.lower(), q.lower()).ratio()
        for q in predefined_answers
    }
    best_q, fuzzy_score = max(fuzzy_scores.items(), key=lambda kv: kv[1], default=("", 0.0))

    if fuzzy_score >= FUZZY_THRESHOLD:
        logger.info(f"✅ Fuzzy override: “{current_q}” ≈ “{best_q}” ({fuzzy_score:.2f}) → predefined answer")
        await send_with_feedback(predefined_answers[best_q], author="Customer Service Agent")
        return
    
        # ─── 4) Prepare retrieval █────────────────────────────────────
    retriever = cl.user_session.get("retriever")
    thread_id = cl.context.session.thread_id
    memory = cl.user_session.get("memory")
    past = memory.get()[-3:]
    context = "\n".join(f"{m.role.title()}: {m.content.strip()}" for m in past if m.content.strip())
    query_with_context = f"{context}\nUser: {message.content}" if context else message.content

    # ─── A) If the user just restarted (via ❌ or 0), clear pre-drill so next input re-prompts ───
    if cl.user_session.get("clarification_just_exited"):
        cl.user_session.set(PRE_DRILL_KEY, False)
        cl.user_session.set(AWAITING_PRE_DRILL, False)
        cl.user_session.set("filtered_nodes", None)
        cl.user_session.set("pre_drill_nodes", None)
        cl.user_session.set("pre_drill_query", None)
        cl.user_session.set(DOC_CHOICES_KEY, None)
        cl.user_session.set("clarification_just_exited", False)

    # ─── 1) Pre-drill: pick the document ───
    if not cl.user_session.get(PRE_DRILL_KEY) and not cl.user_session.get(AWAITING_PRE_DRILL):
        original_q = message.content.strip()
        cl.user_session.set("pre_drill_query", original_q)

        # ─── Use high-K retriever for pre-drill so we get every H3 chunk ───
        import os
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.cohere import CohereEmbedding

        dataset = DATASET_MAPPING.get(cl.user_session.get("chat_profile"))
        vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
        index = VectorStoreIndex.from_vector_store(vector_store)
        pre_drill_retriever = index.as_retriever(
            similarity_top_k=500,
            embedding_model=CohereEmbedding(
                api_key=os.getenv("COHERE_API_KEY"),
                model_name=os.getenv("COHERE_MODEL_ID"),
                input_type="search_document",
                embedding_type="float",
            ),
        )
        all_nodes = pre_drill_retriever.retrieve(original_q)
        cl.user_session.set("pre_drill_nodes", all_nodes)
        
        # — Log each document’s best score —
        doc_scores = {}
        for n in all_nodes:
            src = n.node.metadata.get("source", "Unknown")
            doc_scores[src] = max(doc_scores.get(src, 0.0), n.score)
        for src, score in doc_scores.items():
            logger.info(f"🔍 Doc candidate: '{src}' with top score {score:.3f}")
            
        


        # ─── AUTO-SELECT Policy FAQ.docx if confident ───

        POLICY_AUTO_THRESH    = 0.54   # only policy FAQ ≥0.55 auto-selects
        DOC_CANDIDATE_THRESH  = 0.5   # any doc ≥0.40 is eligible for the user to choose

        policy_score = doc_scores.get("Policy FAQ.docx", 0.0)
        top_score = max(doc_scores.values(), default=0.0)

        if policy_score == top_score and policy_score >= POLICY_AUTO_THRESH:
            # High-confidence hit in Policy FAQ.docx → pick it immediately
            logger.info(f"✅ Auto-selected 'Policy FAQ.docx' (score {policy_score:.3f})")
            filtered = [
                n for n in all_nodes
                if n.node.metadata.get("source") == "Policy FAQ.docx"
            ]
            cl.user_session.set("filtered_nodes", filtered)
            cl.user_session.set(PRE_DRILL_KEY, True)
            # mark that we auto-selected Policy FAQ so we can bypass auto-drill
            cl.user_session.set("policy_auto_select", True)

        else:
            # Fallback: run original multi-doc selection logic
            doc_scores = defaultdict(float)
            for n in all_nodes:
                src = n.node.metadata.get("source", "Unknown")
                doc_scores[src] = max(doc_scores[src], n.score)

            # Only docs ≥ threshold are candidates
            # Only docs ≥ DOC_CANDIDATE_THRESH *and* not the FAQ are candidates
            doc_set = [
                src for src, score in doc_scores.items()
                if score >= DOC_CANDIDATE_THRESH and src != "Policy FAQ.docx"
            ]
            # If that yields ≤1 but you still have multiple docs overall, fall back
            if len(doc_set) <= 1 and len(doc_scores) > 1:
                # prompt on the top 5 by score (excluding FAQ)
                doc_set = [
                    src for src, _ in
                    sorted(doc_scores.items(), key=lambda x: -x[1])
                    if src != "Policy FAQ.docx"
                ][:5]
            for src in doc_set:
                logger.info(f"✅ Candidate doc: '{src}' (score {doc_scores[src]:.3f})")

            if len(doc_set) > 1:
                cl.user_session.set(AWAITING_PRE_DRILL, True)
                cl.user_session.set(DOC_CHOICES_KEY, doc_set)
                options = "\n".join(f"{i+1}. {d}" for i, d in enumerate(doc_set))
                await send_with_feedback(
                    f"❓ คำถามของคุณเกี่ยวกับเนื้อหาหลายเอกสาร โปรดเลือกเอกสารที่ตรงกับความต้องการ:\n\n{options}\n\n"
                    "ตอบด้วยหมายเลข เช่น `1` หรือชื่อเอกสาร",
                    author="Customer Service Agent"
                )
                return

            # Single candidate → auto-pick it
            cl.user_session.set(PRE_DRILL_KEY, True)
            if doc_set:
                single = doc_set[0]
                filtered = [
                    n for n in all_nodes
                    if n.node.metadata.get("source") == single
                ]
                cl.user_session.set("filtered_nodes", filtered)
        # ────────────────────────────────────────────────────────────────────


 

    # ─── 2) Handle the user’s document choice ───
    if cl.user_session.get(AWAITING_PRE_DRILL):
        choice = message.content.strip()
        docs = cl.user_session.get(DOC_CHOICES_KEY) or []
        idx = None

        if choice.isdigit():
            i = int(choice) - 1
            if 0 <= i < len(docs):
                idx = i
        else:
            ratios = [
                SequenceMatcher(None, choice.lower(), d.lower()).ratio()
                for d in docs
            ]
            if ratios and max(ratios) > 0.6:
                idx = ratios.index(max(ratios))

        if idx is None:
            await send_with_feedback("⚠️ โปรดระบุหมายเลขหรือชื่อเอกสารให้ถูกต้องอีกครั้ง")
            return

        selected_doc = docs[idx]
        cl.user_session.set("current_doc", selected_doc)
        cl.user_session.set(PRE_DRILL_KEY, True)
        cl.user_session.set(AWAITING_PRE_DRILL, False)

        # ─── Re-retrieve using high-K retriever so we get every H3 chunk ───
        import os
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.cohere import CohereEmbedding

        dataset = DATASET_MAPPING.get(cl.user_session.get("chat_profile"))
        vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
        index = VectorStoreIndex.from_vector_store(vector_store)
        doc_retriever = index.as_retriever(
            similarity_top_k=500,  # fetch up to 500 chunks
            embedding_model=CohereEmbedding(
                api_key=os.getenv("COHERE_API_KEY"),
                model_name=os.getenv("COHERE_MODEL_ID"),
                input_type="search_document",
                embedding_type="float",
            ),
        )

        # Use the original question so we get all sections of that doc
        query = cl.user_session.get("pre_drill_query") or message.content
        retrieved_nodes = doc_retriever.retrieve(query)

        # Now filter down to just the user-selected document
        filtered = [
            n for n in retrieved_nodes
            if n.node.metadata.get("source") == selected_doc
        ]

        logger.info("📄 User selected doc: %s", selected_doc)
        logger.info("🔍 Retrieved %d chunks for that doc", len(filtered))

        cl.user_session.set("pre_drill_nodes", filtered)
        cl.user_session.set("filtered_nodes", filtered)
        # message.content remains unchanged so H2/H3 logic fires normally
        
    # ─── 2b) Handle hierarchical clarification selection ───
    if cl.user_session.get("awaiting_clarification"):
        level = cl.user_session.get("clarification_level", 2)
        hier  = cl.user_session.get("hier_sections", {})   # { title: [nodes] }
        choice = message.content.strip()

        # Build the options
        titles     = list(hier.keys())
        exit_label = "❌ ถามคำถามใหม่"
        opts       = titles + [exit_label]

        idx = None

        # 1) Digit?
        if choice.isdigit():
            i = int(choice) - 1
            if 0 <= i < len(opts):
                idx = i

        # 2) Exact title?
        if idx is None and choice in titles:
            idx = titles.index(choice)

        # 3) Fuzzy match (ratio > 0.6)
        if idx is None:
            from difflib import SequenceMatcher
            best = (0.0, None)   # (ratio, index)
            for i, t in enumerate(titles):
                r = SequenceMatcher(None, choice, t).ratio()
                if r > best[0]:
                    best = (r, i)
            if best[0] > 0.6:
                idx = best[1]

        # 4) Exit label
        if idx is None and choice == exit_label:
            idx = len(opts) - 1

        # Invalid?
        if idx is None:
            logger.warning(f"⚠️ Invalid hierarchical choice: {choice}")
            await send_with_feedback("⚠️ โปรดเลือกหมายเลขหรือชื่อหัวข้อให้ถูกต้อง")
            return

        selected = opts[idx]
        logger.info(f"🔍 Hierarchical: user picked “{selected}” at level {level}")

        # Exit → restart flow
        if selected == exit_label:
            cl.user_session.set("clarification_just_exited", True)
            return await handle_standard_query(message)

        # Clear menu flags
        cl.user_session.set("awaiting_clarification", False)
        cl.user_session.set("clarification_level", None)
        
        # ─── NEW: top-level H2 pick → only shortcut if no H3 children ───
        if level == 1:
            # look at pre_drill_nodes to see if there are any H3 under this H2
            all_nodes = cl.user_session.get("pre_drill_nodes") or []
            has_h3 = any(
                len(n.node.metadata.get("section_path", [])) >= 3 and
                n.node.metadata["section_path"][1] == selected
                for n in all_nodes
            )
            if not has_h3:
                # no deeper subdivisions → answer immediately on best H2 chunk
                h2_nodes = hier[selected]
                best_h2_chunk = max(h2_nodes, key=lambda n: n.score)
                logger.info(f"✅ H2 “{selected}” has no H3 → immediate answer (score {best_h2_chunk.score:.3f})")
                clear_clarification_state()
                orig_q = cl.user_session.get("pre_drill_query") or message.content
                return await answer_from_node(best_h2_chunk, orig_q)
            # otherwise fall through into your existing H3‐menu logic

        # H2 → show H3 menu
        if level == 2:
            # Grab full pre-drill nodes
            all_nodes = cl.user_session.get("pre_drill_nodes") or []
            from collections import defaultdict
            raw_h3 = defaultdict(list)
            for n in all_nodes:
                path = n.node.metadata.get("section_path", [])
                if len(path) >= 3 and path[1] == selected:
                    raw_h3[path[2]].append(n)

            # No H3 → answer on H2
            if not raw_h3:
                best_h2 = max(
                    (n for n in all_nodes
                     if len(n.node.metadata.get("section_path", [])) >= 2
                     and n.node.metadata["section_path"][1] == selected),
                    key=lambda n: n.score
                )
                clear_clarification_state()
                orig_q = cl.user_session.get("pre_drill_query") or message.content
                return await answer_from_node(best_h2, orig_q)

            # Otherwise show H3 choices
            cl.user_session.set("awaiting_clarification", True)
            cl.user_session.set("clarification_level", 3)
            cl.user_session.set("hier_sections", { h3: raw_h3[h3] for h3 in raw_h3 })

            opts = list(raw_h3.keys()) + [exit_label]
            lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
            logger.info(f"🏷 Showing H3 menu with {len(raw_h3)} options")
            await send_with_feedback(
                "❓ โปรดเลือกหัวข้อย่อย (ระดับ 3):\n\n" + "\n".join(lines),
                author="Customer Service Agent"
            )
            return

        # H3 → answer immediately
        else:  # level == 3
            h3_nodes = hier[selected]
            best_h3   = max(h3_nodes, key=lambda n: n.score)
            logger.info(f"✅ H3 selected: “{selected}” (score {best_h3.score:.3f}) → answer_from_node")
            clear_clarification_state()
            orig_q = cl.user_session.get("pre_drill_query") or message.content
            return await answer_from_node(best_h3, orig_q)



    # ─── 3) Reset on new question ───
    if not cl.user_session.get("awaiting_clarification") and current_q != prev_q:
        cl.user_session.set("auto_skipped", False)
        cl.user_session.set("hier_sections", None)
        cl.user_session.set("clarification_level", None)
        cl.user_session.set("filtered_nodes", None)
    



    # ─── 5) Retrieve (or reuse filtered_nodes) █────────────────────────────────────
    nodes = cl.user_session.get("filtered_nodes")
    if nodes is None:
        try:
            nodes = retriever.retrieve(query_with_context)
            for i, n in enumerate(nodes[:3], 1):
                # grab and clean up the chunk text
                snippet = n.node.get_text().strip().replace("\n", " ")
                # log source, score, and snippet
                logger.info(
                    "🏷 Top #%d: source=%s score=%.3f\n    chunk=\"%s\"",
                    i,
                    n.node.metadata.get("source"),
                    n.score,
                    snippet[:200]  # first 200 chars
                )

            # fuzzy-name fallback...
            name_pattern = r"^[A-Za-zก-๙]+(?:\s+[A-Za-zก-๙]+)+$"
            if re.fullmatch(name_pattern, message.content.strip()):
                best_fuzzy, best_node = 0, None
                for n in nodes:
                    score = SequenceMatcher(None, message.content.strip(), n.node.text.strip()).ratio()
                    if score > best_fuzzy:
                        best_fuzzy, best_node = score, n
                if best_fuzzy >= 0.6:
                    clear_clarification_state()
                    cl.user_session.set("awaiting_clarification", False)
                    return await answer_from_node(best_node, query_with_context)
        except Exception:
            logger.exception("❌ Retrieval failed")
            await send_with_feedback("⚠️ เกิดข้อผิดพลาดในการค้นหา กรุณาลองใหม่อีกครั้ง")
            return

    # ─── 6) Scores, early-reject, auto-drill & auto-answer ─────────────────────
    top_score = nodes[0].score if nodes else 0.0
    logger.info(
        f"🧪 DEBUG | top_score={top_score:.4f}, "
        f"VECTOR_MIN={VECTOR_MIN_THRESHOLD:.4f}, "
        f"VECTOR_MEDIUM={VECTOR_MEDIUM_THRESHOLD:.4f}"
    )
    logger.warning(
        f"🔍 About to check early-reject: top_score={top_score:.4f} vs VECTOR_MIN_THRESHOLD={VECTOR_MIN_THRESHOLD:.4f}"
    )

    # pick the highest-scoring chunk
    # ─── Promote near-top H2 chunks over H1 ───
    H2_OVERRULE_DELTA = 0.01
    # partition by heading level
    h1_nodes = [n for n in nodes if len(n.node.metadata.get("section_path", [])) == 1]
    h2_nodes = [n for n in nodes if len(n.node.metadata.get("section_path", [])) >= 2]

    if h2_nodes and h1_nodes:
        best_h1_node = max(h1_nodes, key=lambda n: n.score)
        best_h2_node = max(h2_nodes, key=lambda n: n.score)
        if best_h2_node.score >= best_h1_node.score - H2_OVERRULE_DELTA:
            best_node = best_h2_node
        else:
            best_node = best_h1_node
    else:
        best_node = max(nodes, key=lambda n: n.score)

    best_path = best_node.node.metadata.get("section_path", [])
    depth = len(best_path)

    # ─── NEW: deepest‐level + confidence + gap shortcut ───
    DEEP_DIRECT_THRESHOLD = 0.60
    DEEP_GAP_THRESHOLD    = 0.055

    # look at your full pre‐drill to see how deep your document actually goes
    all_pre_drill = cl.user_session.get("pre_drill_nodes") or nodes
    max_depth    = max(len(n.node.metadata.get("section_path", [])) for n in all_pre_drill)
    target_depth = max_depth - 1

    # only consider when our best_node is at the deepest H3 level
    if depth >= 3 and depth == target_depth and best_node.score >= DEEP_DIRECT_THRESHOLD:
        # extract the H2 under which best_node lives
        h2_key = best_path[1]

        # 1) gather true H3 siblings (same depth, same parent H2)
        sibling_scores = [
            n.score
            for n in all_pre_drill
            if (
                len(n.node.metadata.get("section_path", [])) == depth
                and n.node.metadata["section_path"][1] == h2_key
            )
        ]

        # 2) fallback: if none (weird), include any chunk under that H2
        if not sibling_scores:
            sibling_scores = [
                n.score
                for n in all_pre_drill
                if (
                    len(n.node.metadata.get("section_path", [])) >= 2
                    and n.node.metadata["section_path"][1] == h2_key
                )
            ]

        sibling_scores.sort(reverse=True)
        top       = sibling_scores[0]
        runner_up = sibling_scores[1] if len(sibling_scores) > 1 else 0.0
        gap       = top - runner_up

        logger.info(f"🏷 DEBUG siblings H3 scores under H2 “{h2_key}”: {sibling_scores}")
        logger.info(
            f"🏷 DEBUG deepest‐level check: depth={depth}, top={top:.3f}, "
            f"runner_up={runner_up:.3f}, gap={gap:.3f} (threshold {DEEP_GAP_THRESHOLD})"
        )

        if gap >= DEEP_GAP_THRESHOLD:
            logger.info(f"🏷 Deepest‐level direct‐answer (gap {gap:.3f} ≥ {DEEP_GAP_THRESHOLD})")
            clear_clarification_state()
            cl.user_session.set("awaiting_clarification", False)
            return await answer_from_node(best_node, query_with_context)
        else:
            logger.info(f"🏷 Gap too small ({gap:.3f} < {DEEP_GAP_THRESHOLD}) → showing H3 menu")
    # ────────────────────────────────────────────────────────

    # (…then falls through into your normal “no-H2s” or “auto‐drill” or H2/H3 menu code…)
    # ─── fallback to H2/H3 menu as before ───

    # ─── Otherwise fall back to your normal H2/H3 menu logic ───

    # 6a) Early‐reject at top levels (depth<3)
    if (
        top_score < VECTOR_MIN_THRESHOLD
        and not cl.user_session.get("awaiting_clarification")
        and depth < 3
    ):
        await send_with_feedback(
            "❌ คำถามไม่เกี่ยวข้อง กรุณาถามใหม่",
            metadata={"difficulty": "Rejected"}
        )
        save_conversation_log(thread_id, message.id, "bot", "Rejected", "Rejected")
        return

    # ─── 6b) Auto‐drill into H3 of the highest‐scoring H2 (skip H2 menu) ───────────
    VECTOR_AUTO_LEVEL3_THRESHOLD = 0.52
    if not cl.user_session.get("awaiting_clarification") and top_score >= VECTOR_AUTO_LEVEL3_THRESHOLD:
        from collections import defaultdict
        # group all chunks by their H2 title
        raw_h2 = defaultdict(list)
        for n in nodes:
            path = n.node.metadata.get("section_path", [])
            if len(path) >= 2:
                raw_h2[path[1]].append(n)

        # pick the top‐scoring H2
        section_scores = {h2: max(n.score for n in grp) for h2, grp in raw_h2.items()}
        top_h2, top_h2_score = max(section_scores.items(), key=lambda x: x[1])
        logger.info(f"🔍 Best H2 candidate: '{top_h2}' with score {top_h2_score:.3f}")

        # only auto‐drill if we're confident
        if top_h2_score >= VECTOR_AUTO_LEVEL3_THRESHOLD:
            # collect all H3 under that H2
            h3_chunks = [
                n for n in nodes
                if len(n.node.metadata.get("section_path", [])) >= 3
                and n.node.metadata["section_path"][1] == top_h2
            ]

            if h3_chunks:
                logger.info(
                    "🏷 Auto-drill into H3 for '%s' → %d chunks",
                    top_h2, len(h3_chunks)
                )

                # set up the hierarchical menu exactly like manual drill
                cl.user_session.set("awaiting_clarification", True)
                cl.user_session.set("clarification_level", 3)

                # map H3 titles → lists of nodes
                h3_map = {}
                for n in h3_chunks:
                    title = n.node.metadata["section_path"][2]
                    h3_map.setdefault(title, []).append(n)
                cl.user_session.set("hier_sections", h3_map)

                # send the H3 menu
                exit_label = "❌ ถามคำถามใหม่"
                opts = list(h3_map.keys()) + [exit_label]
                lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
                await send_with_feedback(
                    "❓ โปรดเลือกหัวข้อย่อย (ระดับ 3):\n\n" + "\n".join(lines),
                    author="Customer Service Agent"
                )
                return

    # 6c) Auto‐answer if extremely confident
    VECTOR_AUTO_DIRECT_THRESHOLD = 0.62 
    if depth >= 2 and top_score >= VECTOR_AUTO_DIRECT_THRESHOLD:
        logger.info(
            "✅ Auto-answer triggered at depth %d (score %.3f)",
            depth, top_score
        )
        clear_clarification_state()
        cl.user_session.set("awaiting_clarification", False)
        return await answer_from_node(best_node, query_with_context)

    # ─── 8) 0th‐drill: hierarchical section drill ─────────────────────────────────
    # … then your H2‐menu logic follows …

    # ─── 8) 0th-drill: hierarchical section drill ─────────────────────────────────
    # ─── 8) 0th-drill: hierarchical section drill ─────────────────────────────────
    # Determine which H1 the best_node lives under
    best_path = best_node.node.metadata.get("section_path", [])
    best_h1 = best_path[0] if len(best_path) >= 1 else None

    # Use the *entire* document’s chunks, not just the filtered subset
    all_doc_nodes = cl.user_session.get("pre_drill_nodes") or []

    # Collect every H2 under that H1, regardless of retrieval score
    raw_h2 = defaultdict(list)
    for n in all_doc_nodes:
        path = n.node.metadata.get("section_path", [])
        if len(path) >= 2 and path[0] == best_h1:
            raw_h2[path[1]].append(n)

    # Debug: log how many chunks each H2 group has
    for h2, grp in raw_h2.items():
        logger.info(
            "🔍 raw_h2['%s'] → %d chunks, top score %.3f",
            h2,
            len(grp),
            max(c.score for c in grp)
        )

    # Build your section_scores map (you may still need it later)
    section_scores = {h2: max(n.score for n in grp) for h2, grp in raw_h2.items()}

    # ─── Always present *all* H2s in document order ─────────────────────────────
    ordered_h2 = list(raw_h2.keys())
    logger.info(f"🔍 ordered_h2 (all doc H2s): {ordered_h2}")

    # ─── 9) Show H2 menu or dive into H3 / answer as before ────────────────────────
    H2_THRESHOLD = 0.2
    MAX_H2_OPTIONS = 5

    # Filter H2s with score >= threshold
    #filtered_h2s = [h for h in section_scores if section_scores[h] >= H2_THRESHOLD]
    filtered_h2s = list(section_scores.keys())

    # Sort and take top N
    ordered_h2 = list(section_scores.keys())[:MAX_H2_OPTIONS]

    logger.info("🔍 ordered_h2 (filtered ≥ %.2f, top %d): %s",
                H2_THRESHOLD,
                MAX_H2_OPTIONS,
                ", ".join(f"'{h}' ({section_scores[h]:.3f})" for h in ordered_h2))

    if len(ordered_h2) > 1:
        logger.info(f"🏷 Showing H2 menu with {len(ordered_h2)} options")
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("hier_sections", {h: raw_h2[h] for h in ordered_h2})
        cl.user_session.set("clarification_level", 1)      # ← ADD THIS
        exit_label = "❌ ถามคำถามใหม่"
        opts = ordered_h2 + [exit_label]
        lines = [f"{i+1}. {h}" for i, h in enumerate(opts)]
        await send_with_feedback(
            f"❓ โปรดเลือกหัวข้อย่อย (ระดับ 2):\n\n" + "\n".join(lines),
            author="Customer Service Agent"
        )
        return


    # ─── 10) Exactly one H2 → drill into H3 (or answer if no children) ──────────
    # ─── 10) Exactly one H2 → show *all* its H3 chunks ───────────────────────────
    if len(ordered_h2) == 1:
        h2_key = ordered_h2[0]
        logger.info("🏷 Single H2 chosen: %s", h2_key)

        # use the full pre_drill_nodes, not the filtered subset
        all_nodes = cl.user_session.get("pre_drill_nodes") or []

        # group every true H3 under that H2
        from collections import defaultdict
        raw_h3 = defaultdict(list)
        for n in all_nodes:
            path = n.node.metadata.get("section_path", [])
            if len(path) >= 3 and path[1] == h2_key:
                raw_h3[path[2]].append(n)

        # if there really are no H3s, fall back on the H2 chunk
        if not raw_h3:
            best_h2_chunk = max(
                (n for n in all_nodes
                 if len(n.node.metadata.get("section_path", [])) >= 2
                 and n.node.metadata["section_path"][1] == h2_key),
                key=lambda n: n.score
            )
            clear_clarification_state()
            cl.user_session.set("awaiting_clarification", False)
            return await answer_from_node(best_h2_chunk, query_with_context)

        # otherwise, show _every_ H3 title you found
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("clarification_level", 2)
        cl.user_session.set("hier_sections", { title: raw_h3[title] for title in raw_h3 })

        exit_label = "❌ ถามคำถามใหม่"
        opts = list(raw_h3.keys()) + [exit_label]
        lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
        await send_with_feedback(
            "❓ โปรดเลือกหัวข้อย่อย (ระดับ 3):\n\n" + "\n".join(lines),
            author="Customer Service Agent"
        )
        return

   

    # ─── 12) LLM answer ─────────────────────────────────────────────
    ctx2 = "\n".join(m.content for m in memory.get()[-3:] if m.role == "user")
    final_q = f"{ctx2}\n{message.content}" if ctx2 else message.content
    lvl = "Hard" if top_score >= VECTOR_MEDIUM_THRESHOLD else "Medium"
    await answer_with_llm(nodes, final_q, lvl, top_score, fuzzy_score)

    # ─── 13) Reset for next new question ─────────────────────────────────
    # ─── 13) Reset for next new question ─────────────────────────────────
    clear_clarification_state()
    for key in [
        "awaiting_clarification",
        PRE_DRILL_KEY,
        AWAITING_PRE_DRILL,
        "pre_drill_nodes",
        "pre_drill_query",
        DOC_CHOICES_KEY,
        "filtered_nodes",
        "hier_sections",
        "clarification_level",
        "policy_auto_select",
        "auto_skipped",
    ]:
        # booleans go to False, lists/queries/data go to None
        if key in (PRE_DRILL_KEY, AWAITING_PRE_DRILL, "awaiting_clarification"):
            cl.user_session.set(key, False)
        else:
            cl.user_session.set(key, None)

async def start_clarification_flow(nodes: list, original_query: str, fuzzy_candidates: list = None):
    """Initiates the clarification process when a query is too broad."""
    # Ensure fuzzy_clarification_rounds is initialized
    cl.user_session.set("clarification_just_exited", False)
    # If we’re not mid‐clarification, clear any leftover hierarchy state
    if not cl.user_session.get("awaiting_hier_clarification"):
        cl.user_session.set("clarification_level", None)
        cl.user_session.set("filtered_nodes", None)
    if fuzzy_candidates:
        current_round = cl.user_session.get("fuzzy_clarification_rounds") or 0
        cl.user_session.set("fuzzy_clarification_rounds", current_round + 1)
        summaries = []
        summary_to_meta = {}
        for q, score in fuzzy_candidates[:MAX_FUZZY_CLARIFY_TOPICS]:
            summaries.append(q)
            summary_to_meta[q] = ("fuzzy", predefined_answers[q], score)
            logger.info(f"🔍 Fuzzy clarification choice: {q} | Score: {score:.2f}")

        opt_out_choice = "❌ ถามคำถามใหม่"
        if opt_out_choice not in summaries:
            summaries.append(opt_out_choice)

        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("clarification_rounds", 0)
        cl.user_session.set("possible_summaries", summaries)
        cl.user_session.set("nodes_to_consider", [])  # Empty list for fuzzy
        cl.user_session.set("summary_to_meta", summary_to_meta)
        cl.user_session.set("original_query", original_query)

        await send_with_feedback(
            content=(
                "❓ คำถามของคุณอาจตรงกับหัวข้อเหล่านี้\n\n"
                + "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
                + '\n\nโปรดตอบกลับด้วยหมายเลขหรือชื่อหัวข้อที่ต้องการ หรือเลือก \"❌ ถามคำถามใหม่\" หากต้องการเริ่มต้นใหม่'
            ),
            author="Customer Service Agent",
        )
        return

    llm = get_llm_settings(cl.user_session.get("chat_profile"))
    summaries, summary_to_meta = [], {}

    nodes_to_summarize = [n for n in nodes[:MAX_TOPICS_BEFORE_CLARIFY] if any(tok in n.node.text for tok in re.findall(r"\w+", original_query))]
    if len(nodes_to_summarize) < 2:
        nodes_to_summarize = nodes[:MAX_TOPICS_BEFORE_CLARIFY]

    for n in nodes_to_summarize:

            # PREPARE BATCHED PROMPT
        truncs = []
        node_map = {}
        for i, n in enumerate(nodes_to_summarize, 1):
            trunc_text = n.node.text[:1000].strip().replace("\n", " ")
            section_title = n.node.metadata.get("section_title", "")
            if section_title:
                trunc_text = f"{section_title}\n{trunc_text}"
            truncs.append(f"({i})\n{trunc_text}")
            node_map[str(i)] = n


        # Add memory history
        memory: ChatMemoryBuffer = cl.user_session.get("memory")
        prior_messages = memory.get()
        history_snippets = "\n".join(f"{m.role.title()}: {m.content.strip()}" for m in prior_messages if m.content.strip())

        batched_prompt = (
            f'ผู้ใช้ถามว่า: "{original_query}"\n\n'
            f"📜 ประวัติการสนทนา:\n{history_snippets}\n\n"
            f"ต่อไปนี้คือเนื้อหาจากหลายเอกสารที่อาจเกี่ยวข้อง:\n\n"
            + "\n\n".join(truncs)
            + "\n\nกรุณาสรุปแต่ละย่อหน้าเป็นหัวข้อย่อยไม่เกิน 10 คำ โดยใช้หมายเลขเดียวกับเนื้อหา เช่น (1) กรณี..., (2) กรณี..., เป็นต้น"
        )

        # CALL LLM ONCE
        resp = llm.chat([ChatMessage(role="user", content=batched_prompt)])
        lines = resp.message.content.strip().splitlines()

        # MAP RESPONSES BACK TO NODES
        summaries = []
        summary_to_meta = {}
        for line in lines:
            match = re.match(r"\(?(\d+)\)?[\.、:]?\s*(.*)", line)
            if match:
                idx, summary = match.groups()
                if idx in node_map and summary not in summary_to_meta:
                    summaries.append(summary)
                    summary_to_meta[summary] = (node_map[idx], node_map[idx].node.text[:1000], node_map[idx].node.metadata.get("source", "UnknownPolicy"))
        # 🧠 Append fuzzy match questions into the clarification loop
        for i, (question, score) in enumerate(fuzzy_candidates, 1):
            label = f'✅ คำถามสำเร็จรูป: "{question}"'
            if label not in summaries:
                summaries.append(label)
                summary_to_meta[label] = ("fuzzy", predefined_answers[question], score)
    opt_out_choice = "❌ ถามคำถามใหม่"
    if opt_out_choice not in summaries:
        summaries.append(opt_out_choice)

    # Set session state for clarification
    cl.user_session.set("awaiting_clarification", True)
    cl.user_session.set("possible_summaries", summaries)
    cl.user_session.set("nodes_to_consider", nodes_to_summarize)
    cl.user_session.set("summary_to_meta", summary_to_meta)
    cl.user_session.set("original_query", original_query)

    # Persist state to DB
    payload = {
        "summaries": summaries,
        "nodes": [{"score": n.score, "text": n.node.text, "meta": n.node.metadata} for n in nodes_to_summarize],
    }
    dl = get_data_layer()
    engine = dl.engine
    async with AsyncSession(engine) as session:
        await session.execute(
            pg_insert(clarification_state).values(thread_id=cl.context.session.thread_id, **payload)
            .on_conflict_do_update(index_elements=["thread_id"], set_=payload)
        )
        await session.commit()

        # 🧠 Log clarification details to terminal
        logger.info("📌 Clarification Triggered")
        logger.info(f"🔍 User Question: {original_query}")
        logger.info("📑 Selected Chunks for Clarification:")
        for i, n in enumerate(nodes_to_summarize):
            preview = n.node.text[:120].replace("\n", " ")
            logger.info(f"  {i+1}. Title: {n.node.metadata.get('section_title', 'Unknown')} | Score: {n.score:.4f} | Preview: {preview}")

        logger.info("🧠 Clarification Choices:")
        for i, s in enumerate(summaries):
            logger.info(f"  {i+1}. {s}")
            if isinstance(summary_to_meta.get(s), tuple) and summary_to_meta[s][0] == "fuzzy":
                logger.info(f"     ↳ Predefined answer score: {summary_to_meta[s][2]:.2f}")

        await send_with_feedback(
            content=(
                "❓ พบเอกสารหลายรายการที่อาจเกี่ยวข้องกับคำถามของคุณ\n\n"
                "หัวข้อที่เป็นไปได้:\n"
                + "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
                + '\n\nโปรดตอบกลับด้วยหมายเลขหรือชื่อหัวข้อที่ต้องการ หรือเลือก "❌ ถามคำถามใหม่" หากต้องการเริ่มต้นใหม่'
            ),
            author="Customer Service Agent",
        )



async def answer_with_llm(nodes: list, query: str, level: str, top_score: float, fuzzy_score: float):
    clear_clarification_state()
    cl.user_session.set("awaiting_clarification", False)
    """Generates an answer using the LLM with context from retrieved nodes."""
    runnable = cl.user_session.get("runnable")
    TOP_K = 3
    selected_nodes = nodes
    logger.info("📤 answer_with_llm contexts (final): %s", [n.node.text for n in selected_nodes])
    contexts = [
        (n.node.metadata.get("source", "Unknown"), n.node.text.strip().replace("\n", " "))
        for n in selected_nodes
    ]

    # Build chunk context
    chunk_context = "".join(
        f'({i}) เอกสารนโยบาย: "{src}"\n'
        f'เนื้อหาชิ้นนี้ (เต็มข้อความ):\n"""{txt}\n"""\n\n'
        for i, (src, txt) in enumerate(contexts, 1)
    )

    # Include prior messages from memory
    memory = cl.user_session.get("memory")
    prior_messages = memory.get()[-5:]
    history_snippets = ""
    for m in prior_messages:
        if m.role == "user":
            history_snippets += f"👤 ผู้ใช้: {m.content.strip()}\n"
        elif m.role == "assistant":
            history_snippets += f"🤖 บอท: {m.content.strip()}\n"

    logger.info("🧠 Chat Memory Used in Prompt:")
    for m in prior_messages:
        logger.info(f"{m.role}: {m.content.strip()}")

    context_str = (
        f"📜 ประวัติการสนทนา:\n{history_snippets}\n\n"
        f"📚 ข้อความจากเอกสาร:\n{chunk_context}"
    )

    # Hint for table formatting if relevant
    suggest_table = any(
        kw in txt for _, txt in contexts
        for kw in ["20 ล้านบาท", "500,000 บาท", "ประเภทที่", "โครงการ"]
    )
    formatting_hint = (
        "\n🧮 คำแนะนำสำคัญ: คำถามของผู้ใช้มีการระบุ 'มูลค่า' ที่ชัดเจน ...\n"
        "หากสามารถจัดให้อยู่ในรูปแบบ **ตาราง Markdown** ได้ ...\n"
    ) if suggest_table else "\nหากเหมาะสม ให้จัดคำตอบในรูปแบบ bullet หรือย่อหน้าเพื่อความเข้าใจง่าย"

    constraint = (
        "\n\n🔒 โปรดใช้เฉพาะข้อมูลจากชิ้นเนื้อหาด้านบนที่ส่งมาเท่านั้น "
        "และอย่าอ้างอิงเนื้อหาในส่วนอื่นๆ"
    )

    filtered_query = (
        f'ผู้ใช้ถามว่า: "{query}"\n\n'
        f"กรุณาตอบโดยอาศัยเนื้อหาต่อไปนี้ทั้งหมด:\n\n{context_str}"
        f"{formatting_hint}"
        f"{constraint}\n\n"
        "ในคำตอบให้ระบุชื่อเอกสารนโยบายที่ใช้ ... ถ้าไม่แน่ใจให้ถามกลับ"
    )

    # ─── Start the thinking animation ───
    animation_task = asyncio.create_task(
        send_animated_message(
            base_msg="กำลังเช็ค Policy ให้อยู่ รอสักครู่นะคะ...",
            frames=["🌑","🌒","🌓","🌔","🌕","🌖","🌗","🌘"],
            interval=0.3
        )
    )

    # ─── Call the LLM off the event loop ───
    try:
        resp = await asyncio.to_thread(runnable.query, filtered_query)
        answer_body = (
            resp.response.strip()
            if hasattr(resp, "response")
            else "".join(resp.response_gen).strip()
        )
    except Exception as e:
        answer_body = f"⚠️ LLM error: {e}"
    finally:
        # ─── Stop the animation ───
        animation_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await animation_task

    answer_body = extract_and_format_table(answer_body)

    # Render a clean markdown table if present
    if "|" in answer_body and "---" in answer_body:
        answer_body = f"**คำตอบ**\n\n{answer_body.strip()}"

    final_answer = (
        f"{answer_body}\n\n"
        f"🧠 *DEBUG* | Category: **{level}** | "
        f"Method: **VectorStore + LLM (top {TOP_K})** | "
        f"Vector: {top_score:.2f} | Fuzzy: {fuzzy_score:.2f}"
    )

    # Send & log
    memory.put(ChatMessage(role="assistant", content=answer_body))
    await send_with_feedback(final_answer, metadata={"difficulty": level})
    save_conversation_log(
        cl.context.session.thread_id,
        cl.context.session.id,
        "bot",
        final_answer,
        level
    )

    # ─── Auto‐clear entire chat memory after LLM reply ───
    thread_id = cl.context.session.thread_id
    user_id   = cl.user_session.get("user").identifier
    redis_key = f"{user_id}:{thread_id}"
    redis_client.delete(redis_key)
    fresh_mem = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT,
        chat_store=chat_store,
        chat_store_key=redis_key,
    )
    cl.user_session.set("memory", fresh_mem)

    # Reset any leftover hierarchical state
    cl.user_session.set("clarification_level", None)
    cl.user_session.set("filtered_nodes", None)
    
# ======================================================================================
# Background Tasks (Admin Replies)
# ======================================================================================

async def poll_all_admin_replies(thread_id: str):
    """Polls Redis for all admin replies in a given thread."""
    printed_keys = set()
    while True:
        try:
            keys = redis_client.keys(f"admin-reply:{thread_id}:*")
            for key in keys:
                key_str = key.decode("utf-8")
                raw = redis_client.get(key)
                if not raw:
                    continue

                payload = json.loads(raw.decode("utf-8"))
                parent_content = payload.get("parent_content", "")
                replies = payload.get("replies", [])
                parent_id = key_str.split(":")[2]
                last_reply_id = replies[-1]["id"] if replies else None

                if shown_admin_replies.get(key_str) == last_reply_id:
                    continue

                if key_str not in printed_keys:
                    await send_with_feedback(f"🧾 Original Question:\n\n{clean_parent_content(parent_content)}", author="User")
                    printed_keys.add(key_str)

                for r in replies:
                    reply_id = r.get("id")
                    if reply_id and not shown_admin_replies.get(f"{key_str}:{reply_id}"):
                        content = r.get("body", {}).get("content", "")
                        cleaned = strip_html(content)
                        if cleaned:
                            await send_with_feedback(f"📬 Reply from Admin:\n\n{cleaned}", author="Admin", parent_id=parent_id)
                            shown_admin_replies[f"{key_str}:{reply_id}"] = True

                if last_reply_id:
                    shown_admin_replies[key_str] = last_reply_id
        except Exception as e:
            logger.error(f"❌ Redis polling error: {e}")
        await asyncio.sleep(5)
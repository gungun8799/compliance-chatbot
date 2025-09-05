
select u."identifier" as employee_id,
       u."metadata" ->> 'email' as email,
       t."id" as sesion_id,
       t."name" as session_name,
       case s."type" 
         when 'user_message' then 'USER'
         when 'assistant_message' then 'ASSISTANT'
         when 'run' then 'RUN'
         else s."type"::text
      end as role,
      to_char(s."createdAt" at time zone 'UTC' at time zone %(timezone)s, 'YYYY-MM-DD HH24:MI:SS น.') as created_time,
      s."output" as message,
      f."value" as feedback_value, 
      f."comment" as feedback_comment 

from "Thread" t

inner join "Step" s  
on (t.id = s."threadId")

inner join "User" u
on (t."userId" = u.id)

left join "Feedback" f 
on (s."id" = f."stepId")

where u."identifier" = %(employee_id)s
      and (s."type" in ('assistant_message', 'user_message') or s."type" = 'run' and not (f."comment" is null or f."value" is null))
      and s."createdAt" between %(start_date)s and %(end_date)s
order by t."id" ASC, s."createdAt" asc
# Chainlit UI Style Development Playground

This is a standalone environment for developing and testing Chainlit UI styles without touching production code or Docker.

## 🚀 Quick Start

### Option 1: Python Server (Recommended)
```bash
cd mock_style
python3 server.py
```
Then open: http://localhost:8888

### Option 2: Direct File Opening
Simply open `index.html` in your browser:
```bash
cd mock_style
open index.html  # Mac
# or
xdg-open index.html  # Linux
# or just double-click the file
```

## 📁 Files Structure

```
mock_style/
├── index.html    # Mock Chainlit UI with production classes
├── theme.css     # YOUR STYLE DEVELOPMENT FILE - EDIT THIS!
├── server.py     # Simple dev server (optional)
└── README.md     # This file
```

## 🎨 Development Workflow

1. **Edit `theme.css`** - This is where you develop your styles
2. **Save the file** - The browser will auto-reload (checks every 1 second)
3. **See changes instantly** in the mock UI
4. **Iterate** until you're happy with the design

## 🔄 CSS Hot-Reload

The HTML includes an auto-reload script that:
- Checks `theme.css` for changes every second
- Automatically refreshes the page when CSS is modified
- No build tools or npm required!

## 📦 Production Handoff

Once you're satisfied with your styles, copy them to production:

### For Production Chainlit App:

1. **Copy CSS to production theme file:**
```bash
# Copy your developed styles to production
cp theme.css ../chainlit_app/theme.css
```

2. **Or copy to public/custom.css:**
```bash
cp theme.css ../chainlit_app/public/custom.css
```

3. **Enable CSS in Chainlit config:**
Edit `chainlit_app/.chainlit/config.toml`:
```toml
[theme]
custom_css_path = "theme.css"

# OR for public folder
[UI]
custom_css = "/public/custom.css"
```

4. **Rebuild Docker container:**
```bash
cd ..
docker compose --env-file .env.dev -p compliance-chatbot down
docker rmi compliance-app:v1.0
docker compose --env-file .env.dev -p compliance-chatbot up --build -d
```

## 🏷️ Important Classes Used

The mock uses the same classes as production Chainlit:

### Layout
- `.app`, `#root` - Main app container
- `.sidebar`, `[data-testid="sidebar"]` - Side navigation
- `.main-container`, `[data-testid="main-container"]` - Main content area
- `.header`, `[role="banner"]` - Top header

### Chat Elements
- `.message` - Message container
- `.message.assistant`, `.bot-message` - Bot messages
- `.message.user`, `.user-message` - User messages
- `.message-content` - Message text content
- `.messages-container` - Messages list

### Input Area
- `.chat-input-container`, `.input-container` - Input wrapper
- `.chat-input`, `.input-field` - Text input
- `.send-button` - Send button

### Interactive Elements
- `.btn`, `.button` - Buttons
- `.nav-item`, `.sidebar-item` - Navigation items

## 🎯 Styling Tips

1. **Brand Colors** - Already defined as CSS variables:
   - `--cp-blue: #0071cd`
   - `--cp-yellow: #ffc01f`
   - `--cp-light-cream: #f3f8ec`
   - `--cp-light-blue: #94c9ea`
   - `--cp-dark-blue: #105584`

2. **Test Different States:**
   - Hover effects on buttons and nav items
   - Focus states on inputs
   - Message styling for both user and assistant

3. **Responsive Design:**
   - Test with browser dev tools
   - Sidebar hides on mobile (< 768px)

## 🧹 Cleanup

When done, you can delete this entire folder:
```bash
cd ..
rm -rf mock_style
```

## 💡 Notes

- No backend required - pure frontend mock
- No npm/node needed - uses browser's native features
- CSS changes reflect immediately (1-second auto-check)
- All production classes are mirrored for 1:1 transfer
- Thai text included in mock for testing Thai fonts

---

Happy styling! 🎨
#!/usr/bin/env python3
"""
Simple HTTP server with auto-reload for CSS development
No dependencies required - uses Python's built-in modules
"""

import http.server
import socketserver
import os

PORT = 8888
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Disable caching for development
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🚀 Development server running at: http://localhost:{PORT}")
        print(f"📁 Serving files from: {DIRECTORY}")
        print(f"🔄 CSS changes will auto-reload in browser")
        print(f"⌨️  Press Ctrl+C to stop")
        httpd.serve_forever()
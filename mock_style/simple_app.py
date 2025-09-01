import chainlit as cl

@cl.on_chat_start
async def start():
    await cl.Message(content="สวัสดีครับ ผม Compliance Policy Assistant ยินดีต้อนรับสู่ระบบตอบคำถามเกี่ยวกับนโยบัย").send()

@cl.on_message
async def main(message: cl.Message):
    await cl.Message(content=f"คุณถาม: {message.content}\n\nนี่คือคำตอบเกี่ยวกับนโยบายการปฏิบัติตามกฎระเบียบ").send()
# app.py
import os
import tempfile
import streamlit as st
from rag1_trained import ChatPDF
import base64
# 初始化核心状态
if "assistant" not in st.session_state:
    st.session_state.assistant = ChatPDF()
if "messages" not in st.session_state:
    st.session_state.messages = []

# 页面配置
st.set_page_config(page_title="LaTeX 生成助手", layout="wide")

# 侧边栏文件上传
with st.sidebar:
    st.header("文件上传(可上传多个文件)")
    
    # PDF上传（支持多文件）
    pdf_files = st.file_uploader(
        "上传PDF文档", 
        type=["pdf"],
        accept_multiple_files=True
    )
    if pdf_files:
        for file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.getvalue())
                st.session_state.assistant.ingest(tmp.name)
            os.unlink(tmp.name)
        st.success(f"已加载 {len(pdf_files)} 个PDF文档")
    
    # 图片上传
    image_file = st.file_uploader("上传参考图片", type=["png", "jpg", "jpeg"])
# 修改图片上传处理部分
if image_file:
    try:
        # 生成完整Data URI格式
        image_bytes = image_file.getvalue()
        mime_type = image_file.type  # 自动获取MIME类型
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        
        # 规范Base64格式
        if len(base64_str) % 4 != 0:
            base64_str += "=" * (4 - len(base64_str) % 4)
        
        st.session_state.image_base64 = base64_str
        st.image(image_bytes, width=200)
        
        # 调试输出
      
        
    except Exception as e:
        st.error(f"图片处理失败: {str(e)}")
        st.session_state.image_base64 = None

# 主界面
st.title("LaTeX图片转代码与注释助手")
st.caption("结合PDF文档内容与图片特征生成LaTeX代码以及相应的注释")

# 显示聊天记录
for msg, is_user in st.session_state.messages:
    if is_user:
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.code(msg, language="latex")

# 用户输入处理
if prompt := st.chat_input("输入您的需求（如：生成表格的LaTeX代码）"):
    if "image_base64" not in st.session_state or not st.session_state.image_base64:
        st.error("请先在侧边栏上传图片")
        st.stop()
    print(st.session_state.image_base64)
    # 添加用户消息
    st.session_state.messages.append((prompt, True))
    
    # 生成响应
    with st.spinner("正在生成LaTeX代码..."):
        try:
            response = st.session_state.assistant.ask(
                query=prompt,
                image_base64=st.session_state.image_base64  # 传递base64数据
            )
            
            st.session_state.messages.append((response, False))
            
            # 显示最新消息
            with st.chat_message("assistant"):
                #st.code(response, language="latex")
                
                st.markdown(response)
                # st.markdown(response)
        
        except Exception as e:
            st.error(f"生成失败: {str(e)}")
            st.session_state.messages.append((f"错误：{str(e)}", False))
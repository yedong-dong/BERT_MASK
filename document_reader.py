# document_reader.py

import os
from docx import Document
import win32com.client as win32
from PyPDF2 import PdfReader


def read_docx_by_pages(file_path):
        # 打开文档
        doc = Document(file_path)
        # 初始化存储所有段落内容的列表
        paragraphs_content = []
        # 遍历文档中的所有段落
        for paragraph in doc.paragraphs:
            # 获取段落文本
            text = paragraph.text
            # 如果段落文本不为空，则添加到列表中
            if text.strip():
                paragraphs_content.append(text)
        return paragraphs_content

    # # 打开文档
    # doc = Document(file_path)
    #
    # # 初始化存储所有页内容的列表
    # pages_content = []
    #
    # # 初始化当前页的内容
    # current_page_content = []
    #
    # # 遍历文档中的所有段落
    # for paragraph in doc.paragraphs:
    #     # 获取段落文本
    #     text = paragraph.text
    #
    #     # 如果段落文本不为空，则添加到当前页的内容中
    #     if text.strip():
    #         current_page_content.append(text)
    #
    #     # 假设每页有10个段落
    #     if len(current_page_content) >= 10:
    #         # 将当前页的内容添加到列表中
    #         pages_content.append("\n".join(current_page_content))
    #
    #         # 重置当前页的内容
    #         current_page_content = []
    #
    # # 将最后一页的内容添加到列表中
    # if current_page_content:
    #     pages_content.append("\n".join(current_page_content))
    #
    # return pages_content


def read_doc_by_pages(file_path):
    # 创建Word应用程序对象
    word = win32.Dispatch("Word.Application")
    word.Visible = False

    # 打开文档
    doc = word.Documents.Open(file_path)

    # 初始化存储所有页内容的列表
    pages_content = []

    # 初始化当前页的内容
    current_page_content = []

    # 遍历文档中的所有段落
    for paragraph in doc.Paragraphs:
        # 获取段落文本
        text = paragraph.Range.Text

        # 如果段落文本不为空，则添加到当前页的内容中
        if text.strip():
            current_page_content.append(text)

        # 假设每页有10个段落
        if len(current_page_content) >= 10:
            # 将当前页的内容添加到列表中
            pages_content.append("\n".join(current_page_content))

            # 重置当前页的内容
            current_page_content = []

    # 将最后一页的内容添加到列表中
    if current_page_content:
        pages_content.append("\n".join(current_page_content))

    # 关闭文档
    doc.Close()

    # 退出Word应用程序
    word.Quit()

    return pages_content


def read_pdf_by_pages(file_path):
    # 打开PDF文件
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)

        # 初始化存储所有页内容的列表
        pages_content = []

        # 遍历PDF文件中的所有页
        for page_number in range(len(reader.pages)):
            # 获取当前页的内容
            page = reader.pages[page_number]
            text = page.extract_text()

            # 将当前页的内容添加到列表中
            pages_content.append(text)

    return pages_content


def read_document_by_pages(file_path):
    # 获取文件扩展名
    file_extension = os.path.splitext(file_path)[1].lower()

    # 根据文件扩展名选择相应的读取函数
    if file_extension == ".docx":
        return read_docx_by_pages(file_path)
    elif file_extension == ".doc":
        return read_doc_by_pages(file_path)
    elif file_extension == ".pdf":
        return read_pdf_by_pages(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
# 定义一个全局白名单列表
white_list = []
def addWhiteList(info):
    """
    将传入的 info 列表中的元素添加到白名单中。

    :param info: list, 要添加到白名单的列表
    """
    if isinstance(info, list):  # 检查 info 是否为列表
        for item in info:
            if item not in white_list:  # 避免重复添加
                white_list.append(item)
    else:
        print("输入必须是一个列表。")


import jieba
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 待生成词云的文本
text = """
Python 是一种广泛使用的高级编程语言，其设计哲学强调代码的可读性和简洁的语法。Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。Python 拥有丰富的库和框架，如 NumPy、Pandas 用于数据分析，Django、Flask 用于 web 开发等。Python 在人工智能、机器学习、数据科学等领域也有着广泛的应用，例如 TensorFlow、PyTorch 等深度学习框架都是基于 Python 开发的。Python 的可移植性很好，可以在多种操作系统上运行，如 Windows、Linux、macOS 等。
"""

# 使用 jieba 对文本进行分词
seg_list = jieba.cut(text, cut_all=False)

# 过滤掉长度为1的词语
filtered_seg_list = [word for word in seg_list if len(word) > 1]

# 将过滤后的词语用空格连接起来
seg_text = " ".join(filtered_seg_list)

# 读取图片作为词云的形状
mask_image = np.array(Image.open('粉黑色系2.jpg'))

# 创建 WordCloud 对象
wordcloud = WordCloud(
    font_path='simhei.ttf',  # 指定中文字体路径
    width=800,  # 词云图的宽度
    height=600,  # 词云图的高度
    background_color='white',  # 背景颜色
    max_words=200,  # 显示的最大单词数
    min_font_size=10,  # 最小字体大小
    max_font_size=100,  # 最大字体大小
    random_state=50,  # 随机状态，用于随机颜色和字体大小等
    mask=mask_image  # 使用图片作为词云的形状
).generate(seg_text)

# 从图片中生成颜色
image_colors = ImageColorGenerator(mask_image)

# 使用 matplotlib 展示词云图
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
plt.axis('off')  # 关闭坐标轴
plt.show()

import jieba


class SearcherScan():
    def __init__(self, title_file):
        with open(title_file, 'r') as f:
            titles = f.read()
        self.title_list = list(set(titles.split()))

    def word_match(self, words, title):
        ifmatch = True
        keyword_ = ' '.join(jieba.cut(words))
        for word in keyword_.split():
            if word != ' ' and word not in title:
                ifmatch = False
        return ifmatch

    def conv_query(self, query):
        query_new_parts = []
        for part in list(jieba.cut(query)):
            if part == '(' or part == ')':
                query_new_parts.append(part)
            elif part in ('and', 'AND', 'or', 'OR', 'NOT', 'not', ' '):
                query_new_parts.append(part.lower())
            else:
                query_new_parts.append(
                    "self.word_match('{}',title)".format(part))
        query_new = ''.join(query_new_parts)
        return query_new

    def highlighter(self, doc, word):
        for part in list(jieba.cut(word)):
            if part not in ('(', ')', 'and', 'AND', 'or', 'OR', 'NOT', 'not', ' '):
                doc = doc.replace(
                    part, '<span style="color:red">{}</span>'.format(part))
        return doc

    def search(self, query):
        query_new = self.conv_query(query)
        print(query_new)
        for title in self.title_list:
            if eval(query_new):
                print(title, query)

query = '苹果 and (芯片 or 高通)' # 找到包含 苹果 或 包含 芯片 或 高通 其中一个的文档
searcher = SearcherScan('./爬虫-新闻标题.txt')
searcher.search(query)

import jieba


class SearcherIIndex():
    """倒排索引文本搜索实现类

    用倒排索引
    利用Python的集合运算，来实现候选结果集之间交、并运算

    Attributes:
        index: 检索使用的倒排索引
        max_id: 当前索引的文档最大ID
        doc_list: 索引文档原文
    """

    def __init__(self, docs_file):
        """初始化，用文件中的文本行构建倒排索引

        Args:
            docs_file:包含带索引文档(文本)的文件名

        """
        self.index = dict()
        self.max_id = 0
        self.doc_list = []

        with open(docs_file, 'r') as f:
            docs_data = f.read()

        # 对读取文档的每一行进度 调用 add_doc
        for doc in docs_data.split('\n'):
            self.add_doc(doc)

    def add_doc(self, doc):
        """向索引中添加新文档

        Args:
            doc:待检索的文档(文本)

        Returns:
            新增文档ID
        """
        self.doc_list.append(doc)

        # 分词之后的 每个单词
        # self.index 倒排索引
        for term in list(jieba.cut(doc)):
            # 构建和更新各Term对应的Posting(集合)
            if term in self.index:
                self.index[term].add(self.max_id)
            else:
                self.index[term] = set([self.max_id])
        self.max_id += 1
        return self.max_id - 1

    def word_match(self, word):
        """从倒排索引中获取包含word的候选文档ID集合

        Args:
            word:待检索的词(短语)

        Returns：
            包含待检索词(短语)的文档ID集合
        """
        result = None
        for term in list(jieba.cut(word)):
            if result is None:
                result = self.index.get(term, set())
            else:
                result = result & self.index.get(term, set())
        if result is None:
            result = set()
        return result

    def conv_query(self, query):
        """将用户的查询转换成用eval可运行、返回结果ID集合的代码段

        Args:
            query:待转换的原始查询字符串

        Returns:
            转换完成可通过eval执行返回ID集合的代码段字符串
        """
        query_new_parts = []
        all_parts = list(jieba.cut(query))
        idx = 0
        cache = ''  # 缓存变量，用于回收分词过程被切开的短语片段
        count_parts = len(all_parts)

        # pdb.set_trace()

        while idx < count_parts:
            if all_parts[idx] == '(' or all_parts[idx] == ')':
                query_new_parts.append(all_parts[idx])
            elif all_parts[idx] == ' ':
                query_new_parts.append(' ')
            elif all_parts[idx] in ('and', 'AND'):
                query_new_parts.append('&')
            elif all_parts[idx] in ('or', 'OR'):
                query_new_parts.append('|')
            elif all_parts[idx] in ('not', 'NOT'):
                query_new_parts.append('-')
            # 被分词切开的短语部分回收至缓存
            elif idx + 1 < count_parts and all_parts[idx + 1] not in (' ', ')'):
                cache += all_parts[idx]
            else:
                query_new_parts.append(
                    "self.word_match('{}')".format(cache + all_parts[idx]))
                cache = ''  # 合并完成清空缓存
            idx += 1
        query_new = ''.join(query_new_parts)
        return query_new

    def highlighter(self, doc, word):
        """用word对doc进行HTML高亮

        Args:
            doc:需要高亮的文档
            word:要进行高亮的关键词(查询)

        Returns:
            返回对关键词(查询)进行高亮的文档
        """
        for part in list(jieba.cut(word)):
            # TODO(CHG):短语高亮需要先分词
            if part not in ('(', ')', 'and', 'AND', 'or', 'OR', 'NOT', 'not', ' '):
                doc = doc.replace(
                    part, '<span style="color:red">{}</span>'.format(part))
        return doc

    def search(self, query):
        """用query进行查询返回结果文档列表

        Args:
            query:用户的(复合)布尔查询字符串

        Returns:
            复合查询要求的(高亮)文档结果列表
        """
        result = []

        query_new = self.conv_query(query)
        print(query_new, eval(query_new))
        for did in eval(query_new):
            result.append([self.doc_list[did]])
        return result

searcher = SearcherIIndex('../asserts/爬虫-新闻标题.txt')
print(searcher.index['孟买'], searcher.index['贫民窟'])

query = '泰山 and (台北 or 高通)'
for doc in searcher.search(query):
    print(doc)
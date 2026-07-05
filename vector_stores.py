from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self, metadata_filter=None):
        """返回向量检索器，方便加入 chain。

        :param metadata_filter: 元数据过滤条件字典，如 {"module": "登录"}，
                                为 None 时不过滤，检索全部用例。
        """
        search_kwargs = {"k": config.similarity_threshold}
        if metadata_filter:
            # Chroma 的 retriever 支持 filter 参数做元数据过滤
            search_kwargs["filter"] = metadata_filter
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("登录功能测试")
    print(res)

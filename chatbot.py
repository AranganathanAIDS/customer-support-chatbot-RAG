from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
import warnings
import pickle
import json
warnings.filterwarnings("ignore")

data = CSVLoader("data\\faq_df.csv")
df1=data.load()

data2=CSVLoader("data\\BankFAQs.csv",encoding='utf-8')
df2=data2.load()

        

faq_df=df1+df2

text_splliter=RecursiveCharacterTextSplitter(chunk_size=1000)
text=text_splliter.split_documents(faq_df)


# import os
# api_key = os.environ.get('API_KEY')


with open(r'C:\Users\LENOVO\Music\Prompt_project\.git\config.json') as f:
    config = json.load(f)
api_key = config['api_key']



# Provide the path to the directory where you want to save the embeddings
save_directory = r"C:\Users\LENOVO\Music\Prompt_project\embeddings.pkl"

# Create an instance of the OpenAIEmbeddings class
embeddings = OpenAIEmbeddings(openai_api_key = api_key)

# Save the embeddings to the specified directory
try:
    with open(save_directory, 'wb') as f:
        pickle.dump(embeddings, f)
    print("OpenAI embeddings saved successfully.")
except Exception as e:
    print(f"Error saving OpenAI embeddings: {e}")



persist_dir="db"

vectordb=Chroma.from_documents(text,embeddings,persist_directory=persist_dir)
print(type(vectordb))
llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',openai_api_key = api_key,temperature = 0)


from langchain.chains import RetrievalQA
retriever_chain = RetrievalQA.from_chain_type(llm,retriever = vectordb.as_retriever(),return_source_documents = True)




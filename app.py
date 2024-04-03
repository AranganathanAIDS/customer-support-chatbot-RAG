from flask import Flask, render_template, request, jsonify
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import pickle
import warnings
import json
warnings.filterwarnings("ignore")


with open(r'C:\Users\LENOVO\Music\Prompt_project\.git\config.json') as f:
    config = json.load(f)
api_key = config['api_key']


#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#

# Provide the path to the file where the embeddings are saved
embeddings_file_path = r"C:\Users\LENOVO\Music\Prompt_project\embeddings.pkl"

# Load the embeddings from the saved file
try:
    with open(embeddings_file_path, 'rb') as f:
        embeddings = pickle.load(f)
    print("OpenAI embeddings loaded successfully.")
except Exception as e:
    print(f"Error loading OpenAI embeddings: {e}")



#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
embeddings = OpenAIEmbeddings(openai_api_key = api_key)

chroma_directory = r"C:\Users\LENOVO\Music\Prompt_project\db"
# Create an instance of the Chroma class
vectordb = Chroma(
    persist_directory=chroma_directory,
    embedding_function=embeddings
)

#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#


llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',openai_api_key = api_key,temperature = 0.7)
retriever_chain = RetrievalQA.from_chain_type(llm,retriever = vectordb.as_retriever(),return_source_documents = True)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def fun():
    data = request.get_json()
    prompt = data['user_message']
    prompt.lower()
    res = retriever_chain(prompt)
    result=res['result']
    response = {'sender': 'chatbot', 'message': result}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
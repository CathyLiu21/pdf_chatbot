import streamlit as st
from googletrans import Translator
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
import os
#load api key lib
from dotenv import load_dotenv
import base64


#Background images add function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
#add_bg_from_local('images.jpeg')  

#sidebar contents

with st.sidebar:
    st.title('🦜️🔗 PDF BASED LLM-LANGCHAIN CHATBOT🤗')
    st.markdown('''
    ## About APP:

    The app's primary resource is utilised to create::

    - [streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ''')

    add_vertical_space(4)
    st.write('💡All about pdf based chatbot 🤗')

load_dotenv()

def main():
    st.header("📄Chat with your pdf file🤗")
    #upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
   
    if pdf is not None:
       st.write(pdf.name)

    openai_api_key = st.sidebar.text_input('OpenAI API Key')
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key.")
        st.stop()
    
    
   # os.environ["OPENAI_API_KEY"] = openai_api_key
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

        #langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        #store pdf name
        store_name = pdf.name[:-4]

        
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            #st.write("Already, Embeddings loaded from the your folder (disks)")
        else:
            #embedding (Openai methods) 
            embeddings = OpenAIEmbeddings()

            #Store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)
            
            #st.write("Embedding computation completed")

        #st.write(chunks)
        
        #Accept user questions/query

        query = st.text_input("Ask questions about related your upload pdf file")
        #st.write(query)

#        if query:
#            docs = vectorstore.similarity_search(query=query,k=3)
#            #st.write(docs)
            
            #openai rank lnv process
#            llm = OpenAI(temperature=0,openai_api_key=openai_api_key)
#            chain = load_qa_chain(llm=llm, chain_type= "stuff")
            
#            with get_openai_callback() as cb:
#                response = chain.run(input_documents = docs, question = query)
#                print(cb)
#            st.write(response)

        if query:
            docs = vectorstore.similarity_search(query=query, k=3)

            # openai rank lnv process
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with st.form("language_selection"):
                st.write("Select language for the answer:")
                # allow users to select any destination language
                dest_lang = st.selectbox(
                    "Select destination language:", list(googletrans_languages.values()))
                submit_button = st.form_submit_button(
                    "Translate and Display Answer")

            if submit_button:
                with get_openai_callback() as cb:
                    response = chain.run(
                        input_documents=docs, question=query)
                    st.write("PDF Chatbot Response:")
                    st.write(response)

                try:
                    translator = Translator()
                    translation = translator.translate(
                        response, src="en", dest=dest_lang)
                    if translation is not None and hasattr(translation, 'text') and translation.text:
                        st.write(
                            f"**Translated Answer ({dest_lang}):** {translation.text}")
                    else:
                        st.error(
                            "Translation failed. Please check your input and try again.")
                except Exception as e:
                    st.error(
                        f"An error occurred during translation: {str(e)}")




if __name__=="__main__":
    main()

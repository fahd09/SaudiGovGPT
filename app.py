import numpy as np
import faiss
import openai
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings


# we use openai embeddings  (best result so far for Arabic text)
# although I need to test other options
model = OpenAIEmbeddings()

# if 'secret' not in st.session_state:
#     st.session_state.secret = os.environ['OPENAI_API_KEY']

# user_secret = st.text_input(label = ":blue[OpenAI API key]", placeholder = "Paste your openAI API key, sk-", type = "password")

# if user_secret:
#     openai.api_key = user_secret
#     st.session_state.secret = user_secret

services = np.load('data/data_2k.npz', allow_pickle=True)['services']
embeddings = np.load('data/embeddings_2k.npz')['embeddings'].astype(np.float32)

# todo: we can save these matrices to optimize inference but not a big issue as it is only done once... 
embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
index = faiss.IndexFlatL2(embeddings_normalized.shape[1])
index.add(embeddings_normalized)

def embed_text(text):    
    return model.embed_query(text)

def embed_query(query_text):
    query_embedding = embed_text(query_text)
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    return query_embedding_normalized

def get_hits(user_input, k = 3):
    text_embedded = np.array(embed_text(user_input)).reshape(1,-1).astype(np.float32)
    distances, indices = index.search(text_embedded, k)    
    # print(f'distances: {distances}')

    if distances[0][0] > 0.4:
        return 'NO RESULTS FOUND', indices, distances
    
    hits = []
    for i, idx in enumerate(indices[0]):
        # NOTE: We have do comment out some parts as the prompt have to fit the max window size.
        #           we can experiment with what best make sense 
        hits.append(
            ('Name:' + str(services[idx][0])
             + '\nURL: ' + str(services[idx][1])
            #  + '\nAgency: ' + str(services[idx][2])
             + '\nDescription: ' + str(services[idx][3])
            #  + '\nCustomers: ' + str(services[idx][4])
            #  + '\nDuration: ' + str(services[idx][5])
            #  + '\nCost: ' + str(services[idx][6])
             + '\nSteps: ' + str(services[idx][7])
             + '\nRequirements: ' + str(services[idx][8])
            #  + '\nSupport: ' + str(services[idx][9])
            #  + '\nDocuments: ' + str(services[idx][11])
             )
            )
    
    hits = '\n\n'.join(hits)
    return hits, indices, distances

st.markdown("""=<style>p, textarea, .stButton, h3, ul {direction: RTL;}</style>""", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center'>🤖🇸🇦 Saudi Gov GPT 🇸🇦🤖</h2>", unsafe_allow_html=True)
st.markdown('''واجهة ذكية تجيب عن الأسئلة المتعلقة بمختلف الخدمات التي تقدمها القطاعات الحكومية في السعودية. اكتب السؤال بشكل مفصل قدر الإمكان حتى نجد الخدمة المناسبة بسهولة.''')

user_input = st.text_area("", placeholder = "اسألني عن أي خدمة حكومية...", key="input")

if st.button("Submit", type="primary"):
    st.markdown("----")
    res_box = st.empty()    
    hits, indices, distances = get_hits(user_input)
    prompt = f'''
        Generate an informative answer for a given question based on the most relevant search result.
        Each result contain a name of the service, a description of what the service is for, 
        a URL to execute the service, the detailed instructions on how to execute the service, 
        the requirements that one should have to be able to execute the service, and 
        finally the support information in case the customer needed any help. 
        You must only use information from the provided result. 
        Use a positive and a welcoming tone. Start the answer by welcoming the user and 
        saying something uplifting. Only use the result that answer the question accurately. 
        If different results refer to different entities, write separate answers for each entity. 
        Make sure you mention the requirements of any given service if they are important. 
        Do not exceed 80 words. You must answer in Arabic (Saudi accent only).        
        
        \n\n\nResults:\n\n{hits}
        \n\n\nQuery:\n{user_input}\n\n
    '''
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',      # we can use davinci too but this one is far cheaper
        max_tokens=500,             # you can change this to get a longer answer
        messages=[{'role': 'user', 'content': prompt}],
        temperature=.5,             
        stream=True
    )
    
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    for chunk in response:
        # save the event response
        collected_chunks.append(chunk)  
        # extract the message
        chunk_message = chunk['choices'][0]['delta'].get('content', '')  
        # save the message
        collected_messages.append(chunk_message)  
        result = "".join(collected_messages).strip()
        result = result.replace("\n", "")        
        res_box.markdown(f'{result}')    

    ## the below code can be used when we don't want to stream...    
    # response = openai.ChatCompletion.create(
    #     temperature=.5,
    #     max_tokens=500,
    #     model="gpt-3.5-turbo", 
    #     messages=[{"role": "user", "content":prompt}]
    #     )
    # result = response['choices'][0]['message']['content']
    # res_box.markdown(f'{result}')

    st.markdown('')    
    with st.container():
        for i, idx in enumerate(indices[0]):
            if len(services[idx][2]) ==0:
                st.markdown(f'- [{services[idx][0]}]({services[idx][1]})')
            else:
                st.markdown(f'- {services[idx][2]}: [{services[idx][0]}]({services[idx][1]})')
    
st.markdown("----")
st.markdown("<p style='text-align:center'>Made with ❤️ by Fahd Alhazmi (@fahd09)</p>", unsafe_allow_html=True)
import os
import shutil
import streamlit as st
import pandas as pd

from ragent import ResumeAgent

# Directory to store uploaded files
UPLOAD_DIR = "./uploaded_files"

st.set_page_config(layout='wide')

# Initialize session state to track uploaded files
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

if 'input_files' not in st.session_state:
    st.session_state.input_files = []

if 'input_text' not in st.session_state:
    st.session_state.input_text = ''

if 'process_click' not in st.session_state:
    st.session_state.process_click = False


def save_uploaded_file(uploaded_file):
    
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Detect changes in uploaded files
def detect_file_changes(new_files):
    changes_detected = False
    new_files_dict = {}

    # Read new files and store content
    for file in new_files:
        new_files_dict[file.name] = file

    # Compare new files with session state
    if st.session_state.uploaded_files != new_files_dict:
        changes_detected = True

    # Update session state with new files
    old_state = st.session_state.uploaded_files.copy()
    st.session_state.uploaded_files = new_files_dict

    return changes_detected, old_state


def display_card(name, data):
    with st.expander(f"{name} - Analysis", expanded=False):
        df = pd.DataFrame(data[name])
        csv = df.to_csv(index = False).encode('utf-8')
        avg_llm_score = df['LLM score'].mean()
        avg_binary_score = df['Binary score'].mean()


        st.write(f"Average LLM Score: {avg_llm_score:.2f}")
        st.write(f"Average Binary Score: {avg_binary_score:.2f}")
        st.download_button(
            label = ">Download as csv file", 
            data = csv, 
            file_name = f"{name.split('.')[0]}.csv",
            mime = 'text/csv'
        )
        st.table(df)



# Function to display and update the stream
def display_stream(generator_fn, *args, **kwargs):
    # Streamlit session state to keep track of the data
    st.session_state.data = {}
    st.session_state.current_name = None

    placeholder_name = st.empty()
    
    placeholder_table = st.empty()

    generator = generator_fn(*args, **kwargs)
    
    for name, row in generator:
        if(name != st.session_state.current_name):
            if(st.session_state.current_name is not None):
                display_card(st.session_state.current_name, st.session_state.data)

            st.session_state.current_name = name
            st.session_state.data[name] = []
        
        st.session_state.data[name].append(row)
        # Display the streaming table
        placeholder_name.write(name)
        placeholder_table.table(pd.DataFrame(st.session_state.data[name]))

    placeholder_name.empty()
    placeholder_table.empty()

    display_card(st.session_state.current_name, st.session_state.data)  


def cleanup():
    shutil.rmtree(UPLOAD_DIR)
    

# Streamlit app
st.title("Resume Analyzer")

# File uploader

def handle_file_uploads():
    input_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if input_files:
        st.session_state.input_files.extend(input_files)

# Input Job Description
def handle_input_text():
    input_text = st.text_area("Enter Job Description")

    if input_text:
        st.session_state.input_text = input_text


handle_file_uploads()


handle_input_text()


# Process button
if st.button("Process"):
    input_files = st.session_state.input_files
    job_description = st.session_state.input_text

    if input_files and job_description:
        changes_detected, _ = detect_file_changes(input_files)
        if(changes_detected):
            if os.path.isdir(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
            
        with st.spinner("Processing..."):
            pdf_texts = []
            # Create the directory if it doesn't exist
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)

            for uploaded_file in input_files:
                # Save uploaded file to the directory
                file_path = save_uploaded_file(uploaded_file)
            

            try:
                r_agent = ResumeAgent(UPLOAD_DIR)
                
                display_stream(r_agent.analyze, job_description)  

                del r_agent

                cleanup()

            except Exception as e:
                st.error(str(e))
                cleanup()
        
    else:
        st.error("Please upload PDF files and enter Job Description.")
        cleanup()



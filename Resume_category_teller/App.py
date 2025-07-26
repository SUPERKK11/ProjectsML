import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
knn = pickle.load(open('Resume_category_teller/model.pkl', 'rb'))  # Example file name, adjust as needed
tfidf = pickle.load(open('Resume_category_teller/tfidf.pkl', 'rb'))  # Example file name, adjust as needed
le = pickle.load(open('Resume_category_teller/label_encoder.pkl', 'rb'))  # Example file name, adjust as needed

def cleanResume(text):
    # Remove URLs
    cleaned_text = re.sub('http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # remove @mentions
    cleaned_text = re.sub('@\S+', '', cleaned_text, flags=re.MULTILINE)
    # remove the hashtags
    cleaned_text = re.sub('#\S+', '', cleaned_text, flags=re.MULTILINE)
    # remove RT|CC 
    cleaned_text = re.sub('RT|CC\S+', '', cleaned_text, flags=re.MULTILINE)
    # remove special characters
    cleaned_text = re.sub('[%s]' % re.escape("""!@#$%^&*()-_=+{}[]:;'"?/\><.,~`*-.|"""),'', cleaned_text, flags=re.MULTILINE)
    #remove other unwanted things
    cleaned_text = re.sub(r'[^\x00-\x7f]', '', cleaned_text) 
    # remove extra spaces
    cleaned_text = re.sub('\s+', ' ', cleaned_text).strip()
    #finally return the cleaned text
    return cleaned_text


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = knn.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Make prediction
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
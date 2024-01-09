from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import docx
import textract
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'docx', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_text_from_pdf(file_path):
    text = textract.process(file_path, method='pdftotext')
    return text.decode('utf-8')

def calculate_similarity(resumes, job_descriptions):
    similarity_scores = []

    for resume_text in resumes:
        scores = []
        for job_description_text in job_descriptions:
            vectorizer = CountVectorizer().fit_transform([resume_text, job_description_text])
            vectors = vectorizer.toarray()
            similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
            scores.append(similarity)
        similarity_scores.append(scores)

    return similarity_scores

def format_output(job_name, top_resumes):
    result = [f'{job_name}:']
    for i, (resume_name, similarity) in enumerate(top_resumes):
        result.append(f'{i + 1}. {resume_name}, {round(similarity * 100, 2)}%.')
    return '\n'.join(result)

def rank_resumes(similarity_scores, resumes, job_descriptions, top_count=3):
    # Dapatkan indeks dari top_count skor tertinggi untuk setiap resume
    top_indices = [sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:top_count] for scores in similarity_scores]

    # Mengumpulkan nama resume
    job_names = [os.path.splitext(job_desc["filename"])[0] for job_desc in job_descriptions]

    # Format output untuk setiap resume
    result = [format_output(job_name, [(os.path.splitext(resumes[index])[0], similarity_scores[j][index]) for index in indices]) for j, (job_name, indices) in enumerate(zip(job_names, top_indices))]

    # Menggabungkan hasil menjadi satu daftar
    flat_result = '\n\n'.join(result)
    # print(flat_result)
    return flat_result
    

@app.route('/upload', methods=['POST'])
def upload_files():
    resume_files = request.files.getlist('resumes')
    job_description_files = request.files.getlist('job_descriptions')

    # print(f"Jumlah Resume Files: {len(resume_files)}")
    print(f"Jumlah Job Description Files: {len(job_description_files)}")

    resumes = []
    job_descriptions = []

    # Membaca teks dari file-file resume
    for resume_file in resume_files:
        if resume_file and allowed_file(resume_file.filename):
            filename = secure_filename(resume_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(file_path)
            text = read_text_from_docx(file_path) if filename.endswith('.docx') else read_text_from_pdf(file_path)
            resumes.append({'filename': filename, 'text': text})  # Memperbarui agar menyimpan informasi nama file

    # Membaca teks dari file-file job description
    for job_description_file in job_description_files:
        if job_description_file and allowed_file(job_description_file.filename):
            filename = secure_filename(job_description_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            job_description_file.save(file_path)
            text = read_text_from_docx(file_path) if filename.endswith('.docx') else read_text_from_pdf(file_path)
            job_descriptions.append({'filename': filename, 'text': text})  # Memperbarui agar menyimpan informasi nama file

    # Menghitung similarity scores
    similarity_scores = calculate_similarity([job_desc['text'] for job_desc in job_descriptions], [resume['text'] for resume in resumes])
    
    # Mendapatkan top 3 resume dengan skor tertinggi untuk setiap job description
    top_resumes = rank_resumes(similarity_scores, [resume['filename'] for resume in resumes], job_descriptions)

    return jsonify({'top_resumes': top_resumes})

if __name__ == '__main__':
    app.run()
    # app.run(host='45.9.191.89', port=5000, debug=True)

import os
import re
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import PyPDF2
import pdfplumber
import docx
import spacy
import magic
from typing import Optional, List, Dict, Any

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Constants
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
NAME_PATTERNS = [
    r'(?:name|full name)\s*[:=-]\s*(.+)',
    r'^([A-Z][a-z]+ [A-Z][a-z]+)$'
]

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

class FileProcessor:
    @staticmethod
    def allowed_file(filename: str) -> bool:
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @staticmethod
    def extract_text(filepath: str) -> Optional[str]:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(filepath)

        if file_type == 'application/pdf':
            return FileProcessor._extract_text_from_pdf(filepath)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return FileProcessor._extract_text_from_docx(filepath)
        return None

    @staticmethod
    def _extract_text_from_pdf(filepath: str) -> str:
        text = ""
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed: {e}")
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        return FileProcessor._clean_text(text)

    @staticmethod
    def _extract_text_from_docx(filepath: str) -> str:
        doc = docx.Document(filepath)
        return "\n".join(para.text for para in doc.paragraphs if para.text)

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r'\n\s*\n+', '\n', text.strip())


class ResumeParser:
    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        text = ResumeParser._clean_text(text)
        return {
            'name': ResumeParser._extract_name(text),
            'email': ResumeParser._extract_email(text),
            'cgpa': ResumeParser._extract_cgpa(text),
            'skills': ResumeParser._extract_skills(text),
            'education': ResumeParser._extract_education(text),
            'projects': ResumeParser._extract_projects(text),
            'experience': ResumeParser._extract_experience(text),
            'text': text
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _extract_name(text: str) -> str:
        """Extract name with multiple fallbacks, returning first two words"""
        # Method 1: Explicit name patterns
        for pattern in NAME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match and ResumeParser._is_likely_name(match.group(1)):
                name = match.group(1).strip()
                return ' '.join(name.split()[:2])

        # Method 2: Likely name lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if ResumeParser._is_likely_name(line):
                return ' '.join(line.split()[:2])

        # Method 3: NLP with filtering
        doc = nlp(text)
        for ent in doc.ents:
            if (ent.label_ == "PERSON" and 
                ResumeParser._is_likely_name(ent.text) and 
                len(ent.text.split()) >= 2):
                return ' '.join(ent.text.split()[:2])

        # Method 4: First valid-looking name
        for line in lines:
            words = line.split()
            if len(words) >= 2 and words[0][0].isupper() and words[1][0].isupper():
                return ' '.join(words[:2])

        return "Not Found"

    @staticmethod
    def _is_likely_name(text: str) -> bool:
        """Check if text looks like a person's name"""
        words = text.split()
        if len(words) < 2:
            return False
        return all(word[0].isupper() for word in words[:2])

    @staticmethod
    def _extract_email(text: str) -> str:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else "Not Found"

    @staticmethod
    def _extract_cgpa(text: str) -> str:
        patterns = [
            r'CGPA\s*[:=\-]?\s*(\d\.\d{1,2})',
            r'GPA\s*[:=\-]?\s*(\d\.\d{1,2})',
            r'(\d\.\d{1,2})\s*\(?CGPA\)?',
            r'(\d\.\d{1,2})\s*\(?GPA\)?',
            r'CGPA\s*\/\s*(\d\.\d{1,2})',
            r'Cumulative GPA\s*[:=\-]?\s*(\d\.\d{1,2})'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return "Not Found"

    @staticmethod
    def _extract_skills(text: str) -> str:
        skills = [
            'python', 'java', 'c++', 'javascript', 'html', 'css', 'react', 'angular',
            'node.js', 'express', 'django', 'flask', 'spring', 'machine learning',
            'data analysis', 'sql', 'mongodb', 'postgresql', 'aws', 'docker',
            'kubernetes', 'git', 'rest api', 'graphql', 'tensorflow', 'pytorch',
            'pandas', 'numpy', 'scikit-learn', 'tableau', 'power bi', 'linux',
            'bash', 'php', 'ruby', 'rails', 'swift', 'kotlin', 'android', 'ios',
            'cybersecurity', 'networking', 'blockchain', 'solidity', 'rust', 'go'
        ]
        found_skills = []
        for skill in skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                found_skills.append(skill.title())
        return ', '.join(found_skills) if found_skills else "Not Found"

    @staticmethod
    def _extract_education(text: str) -> str:
        degree_patterns = [
            r'\bB\.?[A-Za-z]\.?\b',
            r'\bM\.?[A-Za-z]\.?\b',
            r'\bPh\.?D\.?\b',
            r'\bBachelor\b',
            r'\bMaster\b',
            r'\bDiploma\b',
            r'\bB\.?Tech\b',
            r'\bB\.?E\.?\b',
            r'\bB\.?Sc\b',
            r'\bM\.?Sc\b'
        ]
        
        education_keywords = ['education', 'academic background', 'qualifications']
        lines = text.split('\n')
        education = []
        in_education_section = False
        
        for line in lines:
            if any(keyword in line.lower() for keyword in education_keywords):
                in_education_section = True
                continue
            
            if in_education_section:
                if not line.strip():
                    in_education_section = False
                    continue
                
                for pattern in degree_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        education.append(line.strip())
                        break
        
        if not education:
            doc = nlp(text)
            for sent in doc.sents:
                for pattern in degree_patterns:
                    if re.search(pattern, sent.text, re.IGNORECASE):
                        education.append(sent.text.strip())
                        break
        
        return "\n".join(education[:3]) if education else "Not Found"

    @staticmethod
    def _extract_projects(text: str) -> str:
        project_keywords = ['project', 'personal project', 'academic project']
        lines = text.split('\n')
        projects = []
        in_project_section = False
        
        for line in lines:
            if any(keyword in line.lower() for keyword in project_keywords):
                in_project_section = True
                continue
            
            if in_project_section:
                if not line.strip():
                    in_project_section = False
                    continue
                
                if (len(line.strip()) > 30 and 
                    not any(word in line.lower() for word in ['skills', 'experience', 'education'])):
                    projects.append(line.strip())
        
        if not projects:
            for sent in nlp(text).sents:
                if 'project' in sent.text.lower() and len(sent.text) > 30:
                    projects.append(sent.text.strip())
        
        return "\n\n".join(projects[:3]) if projects else "Not Found"

    @staticmethod
    def _extract_experience(text: str) -> str:
        exp_keywords = ['experience', 'work history', 'employment history']
        date_pattern = r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b\d{4}\b|\b(?:Present|Current)\b)'
        lines = text.split('\n')
        experience = []
        in_exp_section = False
        
        for line in lines:
            if any(keyword in line.lower() for keyword in exp_keywords):
                in_exp_section = True
                continue
            
            if in_exp_section:
                if not line.strip():
                    in_exp_section = False
                    continue
                
                if (re.search(date_pattern, line) or 
                    any(word in line.lower() for word in [' at ', ' intern ', 'developer', 'engineer', 'analyst'])):
                    experience.append(line.strip())
        
        if not experience:
            doc = nlp(text)
            for sent in doc.sents:
                if (' at ' in sent.text or ' intern ' in sent.text.lower() or 
                    any(word in sent.text.lower() for word in ['developer', 'engineer', 'analyst'])):
                    experience.append(sent.text.strip())
        
        return "\n\n".join(experience[:3]) if experience else "Not Found"


@app.route('/')
def index():
    return render_template('indexy.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not (file and FileProcessor.allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        text = FileProcessor.extract_text(filepath)
        if text is None:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        result = ResumeParser.parse(text)
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, render_template, request
import PyPDF2
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    text = ''
    pdf_reader = PyPDF2.PdfReader(pdf_path)

    num_pages = len(pdf_reader.pages)
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text



nlp = spacy.load('en_core_web_sm')


def parse_resume(text):
    doc = nlp(text)

    # Extract entities
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = [ent.text]
        else:
            entities[ent.label_].append(ent.text)

    return entities



def classify_information(text):
    categories = ['Personal', 'Skills', 'Education', 'Experience', 'Other']

    training_data = [
        ('programming languages, databases', 'Skills'),
        ('Excellent communication and problem-solving skills', 'Skills'),
        ('Proficient in languages like JavaScript, Python, Java, C++, C#,TypeScript, PHP, Swift, Ruby, Go', 'Skills'),
        ('Strong analytical skills and attention to detail', 'Skills'),
        ('Excellent communication and problem-solving skills', 'Skills'),
        ('Bachelor of Engineering in Computer Science', 'Education'),
        ('Master of Science in Computer Science, Any University, 2023', 'Education'),
        ('Bachelor of Arts in English Literature, Major College, 2020', 'Education'),
        ('Studied at National School', 'Education'),
        ('5 years experience as a software developer', 'Experience'),
        ('5 years of experience as a Software Engineer at Tech Company', 'Experience'),
        ('Internship as a Marketing Associate at Marketing Agency, Summer 2022', 'Experience'),
        ('Led a team of developers in building a new e-commerce platform', 'Experience'),
        ('Name, John, Hello', 'Personal'),
        ('238059238', 'Personal'),
        ('Pine street address', 'Personal'),
        ('johndoe@email.com', 'Personal'),
        ('janesmith@company.edu.in', 'Personal'),
        ('Florida, Minessota, Cuba, Earth, location, neighbourhood', 'Personal')
    ]

   
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform([x[0] for x in training_data])
    y_train = [x[1] for x in training_data]

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Classifying the input text
    X_test = vectorizer.transform([text])
    category = classifier.predict(X_test)
    return categories[categories.index(category)]

final_text = ''


employee_resume_1_text = "John Doe [Email Address] | [Phone Number] | [LinkedIn URL (Optional)] Summary Highly skilled and motivated Software Engineer with X years of experience designing, developing, and implementing innovative software solutions. Proven ability to work effectively in a fast-paced environment, collaborate across teams, and deliver high-quality code. Passionate about building user-centric applications and staying current with the latest technologies. Skills Programming Languages: Python (Expert), Java (Proficient), Go (Skilled), C++ (Familiar) Frameworks and Libraries: TensorFlow, PyTorch, Scikit-learn, Spring Boot, Django Cloud Platforms: Google Cloud Platform (GCP), AWS (Experience) Version Control Systems: Git (Expert), SVN (Familiar) Development Methodologies: Agile (Scrum, Kanban), Waterfall Tools: JIRA, Confluence, Docker, Kubernetes (Basic Understanding) Experience Software Engineer | Google | [City, State] | [Start Date - End Date (or Present)] Designed, developed, and deployed a critical microservice using Python and Go for the [Product/Service Name] platform, resulting in a 20% reduction in response time.Collaborated with cross-functional teams (product, design, QA) to implement new features and functionalities for the [Product/Service Name] application, ensuring a seamless user experience. Developed and maintained unit and integration tests using a TDD (Test-Driven Development) approach, fostering high code quality and maintainability. Contributed to the improvement of internal developer tools by creating reusable Python libraries, enhancing developer productivity. Stayed up-to-date with the latest advancements in AI/ML by attending internal Google workshops and conferences. Software Engineer | [Previous Company Name] | [City, State] | [Start Date - End Date] Developed and maintained a web application using Java and Spring Boot framework for [Brief Description of Project]. Implemented unit and integration tests using JUnit framework to ensure code reliability. Collaborated with designers to translate UI/UX mockups into functional code, adhering to best practices and accessibility guidelines. Utilized Git for version control and code collaboration. Education Master of Science in Computer Science | [University Name] | [City, State] | [Graduation Year] Relevant Coursework: Machine Learning, Artificial Intelligence, Distributed Systems, Software Engineering Principles, Algorithms and Data Structures. Bachelor of Science in Computer Science | [University Name] | [City, State] | [Graduation Year]"
final_employee_1_text = ""
entities = parse_resume(employee_resume_1_text)
for entity_type, values in entities.items():
    for value in values:
        if isinstance(value, str):  # Check if value is a string
            category = classify_information(value)
            if category == "Education" or category == "Skills" or category == "Experience":
              print(f'{entity_type} - {value}: {category}')

              final_employee_1_text += (value + " ")
        else:
            print(f'Invalid value for {entity_type}: {value}')



app = Flask(__name__)

def calculate_chance_score(resume_text, employee_text):
  documents = [resume_text, employee_text]

  vectorizer = TfidfVectorizer()

  tfidf_matrix = vectorizer.fit_transform(documents)


  user_vector = tfidf_matrix[0] 
  employee_similarities = cosine_similarity(user_vector.reshape(1, -1), tfidf_matrix[1:])
  scoree = None
  for i, score in enumerate(employee_similarities[0]):
      #print(f"Similarity with Employee {i+1}: {score}")
      scoree = score
  return scoree 


@app.route("/", methods=["GET", "POST"])
def analyze_resume():
  chance_score = 0
  if request.method == "POST":
    # Access uploaded file
    resume_file = request.files["resume"]
    if resume_file and allowed_file(resume_file.filename):
      extracted_text = extract_text_from_pdf(resume_file)
      resume_text = extracted_text

      final_text = ''

      entities = parse_resume(resume_text)
      for entity_type, values in entities.items():
        for value in values:
          if isinstance(value, str):  # Check if value is a string
            category = classify_information(value)
            if category == "Education" or category == "Skills":
              final_text += (value + " ")
          else:
            print(f'Invalid value for {entity_type}: {value}')

      resume_text = final_text
      chance_score = calculate_chance_score(resume_text, final_employee_1_text)
  return render_template("index.html", chance_score=round(chance_score*100))

def allowed_file(filename):
  return filename.lower().endswith(".pdf")

if __name__ == "__main__":
  app.run(debug=True)





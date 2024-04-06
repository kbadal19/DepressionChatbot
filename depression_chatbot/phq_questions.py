

from phq_9.models import PHQ9Question

def populate_questions():
    questions = [
        "Do you have little interest or pleasure in doing things?",
        "Do you often feel down, depressed, or hopeless?",
        "Do you have trouble falling or staying asleep, or do you sleep too much?",
        "Do you frequently feel tired or have little energy?",
        "Do you experience poor appetite or overeating?",
        "Do you feel bad about yourself, or do you feel like you are a failure or have let yourself or your family down?",
        "Do you have trouble concentrating on things, such as reading the newspaper or watching television?",
        "Do you find yourself moving or speaking so slowly that other people could notice, or do you experience the opposite, being so fidgety or restless that you move around a lot more than usual?",
        "Do you have thoughts that you would be better off dead or of hurting yourself in some way?"
    ]
    
    for text in questions:
        question = PHQ9Question(question_text=text)
        question.save()

if __name__ == "__main__":
    populate_questions()

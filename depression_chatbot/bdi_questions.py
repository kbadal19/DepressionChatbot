
from bdi_2.models import BDI2Question

def populate_questions():
    questions = [
        "How often do you feel sad, whether it's a passing feeling or a persistent emotion?",
        
        "Do you often find yourself feeling discouraged about what the future holds for you?",
        
        "Have you experienced failures or setbacks recently, either in your personal or professional life?",
        
        "Are you finding that activities or hobbies you used to enjoy no longer bring you the same level of pleasure or satisfaction?",
        
        "Do you frequently experience feelings of guilt, whether it's related to specific events or a general sense of wrongdoing?",
        
        "Do you feel like you're being punished for something, either by yourself or by external forces?",
        
        "Are you disappointed in yourself or dissatisfied with who you are as a person?",
        
        "Do you tend to blame yourself excessively for mistakes or shortcomings?",
        
        "Have you been having thoughts of harming yourself or ending your life?",
        
        "Do you find yourself crying more often than usual, even over small things?",
        
        "Are you feeling more restless, agitated, or on edge than usual?",
        
        "Have you lost interest in activities or hobbies that used to bring you joy, or do you find it difficult to connect with others?",
        
        "Are you having trouble making decisions, whether they're big life choices or simple daily tasks?",
        
        "Do you often feel worthless or have a low opinion of yourself?",
        
        "Do you frequently feel tired or lacking in energy, even after getting enough rest?",
        
        "Have you noticed changes in your sleeping patterns, such as difficulty falling asleep or sleeping more than usual?",
        
        "Are you easily irritated or frustrated by minor things that wouldn't usually bother you?",
        
        "Have you experienced changes in your appetite, such as eating more or less than usual?",
        
        "Do you find it challenging to concentrate or focus on tasks, even ones that used to be easy for you?",
        
        "Are you feeling physically exhausted or fatigued, regardless of how much rest you get?",
        
        "Have you lost interest in sexual activity or experienced a decrease in your libido?",
    ]
    
    for text in questions:
        question = BDI2Question(question_text=text)
        question.save()

if __name__ == "__main__":
    populate_questions()


grades = [4, 73, 67, 38, 33]

def gradingStudents(grades):
    scores = grades[1:]
    for score in scores:
        if score < 38:
            print(score)
        elif (score % 5) <= 3:
            print(score + (5 - (score % 5)))
        else:
            print(score)

gradingStudents(grades)

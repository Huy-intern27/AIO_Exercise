class Person():
    def __init__(self, name, yob):
        self._name = name
        self._yob = yob

    def get_yob(self):
        return self._yob

    def describe_person(self):
        return f'Name: {self._name} - YoB: {self._yob}'

class Student(Person):
    def __init__(self, name, yob, grade):
        super().__init__(name, yob)
        self.grade = grade

    def describe(self):
        print(f'Student - {self.describe_person()} - Grade: {self.grade}')

class Teacher(Person):
    def __init__(self, name, yob, subject):
        super().__init__(name, yob)
        self.subject = subject

    def describe(self):
        print(f'Teacher - {self.describe_person()} - Subject: {self.subject}')

class Doctor(Person):
    def __init__(self, name, yob, specialist):
        super().__init__(name, yob)
        self.specialist = specialist

    def describe(self):
        print(f'Doctor - {self.describe_person()} - Specialist: {self.specialist}')

class Ward():
    def __init__(self, name):
        self.name = name
        self.list_ward = []

    def add_person(self, person):
        self.list_ward.append(person)

    def describe(self):
        print(f'Ward Name: {self.name}')
        for person in self.list_ward:
            person.describe()

    def count_doctor(self):
        count = 0
        for person in self.list_ward:
            if isinstance(person, Doctor):
                count += 1
        return count

    def sort_age(self):
        self.list_ward.sort(key = lambda person : -person.get_yob())

    def compute_average(self):
        sum_yob = 0
        count_teacher = 0
        for person in self.list_ward:
            if isinstance(person, Teacher):
                sum_yob += person.get_yob()
                count_teacher += 1
        return sum_yob / count_teacher

if __name__ == "__main__":
    student1 = Student("studentA", 2010, "7")
    teacher1 = Teacher("teacherA", 1969 , "Math")
    doctor1 = Doctor ("doctorA", 1945 , "Endocrinologists")
    teacher2 = Teacher("teacherB", 1995 , "History")
    doctor2 = Doctor("doctorB",1975 , "Cardiologists ")

    ward1 = Ward(name = "Ward1")
    ward1.add_person(student1 )
    ward1.add_person(teacher1 )
    ward1.add_person(teacher2 )
    ward1.add_person(doctor1 )
    ward1.add_person(doctor2 )
    ward1.describe()

    print(f"\nNumber of doctors : {ward1.count_doctor()}")

    print ("\nAfter sorting Age of Ward1 people ")
    ward1.sort_age()
    ward1.describe()

    print(f"\nAverage year of birth (teachers ): {ward1.compute_average()}")
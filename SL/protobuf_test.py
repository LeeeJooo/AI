from msg_pb2 import Person

# 메시지 생성 및 필드 설정
person = Person(name="John Doe", id=1234, email="jdoe@example.com")

# 메시지 직렬화
serialized_data = person.SerializeToString()

# 메시지 역직렬화
new_person = Person()
new_person.ParseFromString(serialized_data)

print(new_person)
print(serialized_data)

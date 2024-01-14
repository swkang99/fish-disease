edward = {
    '에드워드체표증상': False,
    # '에드워드복부증상': False,
    # '에드워드항문증상': False,
    '에드워드복수증상': False,
    '에드워드간증상': False,
    # '에드워드신장증상': False,
    # '에드워드비장증상': False,
    '에드워드장증상': False
}

vibrio = {
    '비브리오체표증상': False,
    # '비브리오입증상': False,
    '비브리오지느러미증상': False,
    '비브리오아가미증상': False,
    '비브리오간증상': False,
    # '비브리오신장증상': False,
    # '비브리오비장증상': False
}

strepto = {
    '연쇄구균체표증상': False,
    # '연쇄구균복부증상': False,
    # '연쇄구균눈증상': False,
    '연쇄구균아가미뚜껑증상': False,
    '연쇄구균지느러미증상': False,
    # '연쇄구균항문증상': False,
    '연쇄구균아가미증상': False,
    '연쇄구균복수증상': False,
    '연쇄구균간증상': False,
    # '연쇄구균신장증상': False,
    # '연쇄구균비장증상': False,
    '연쇄구균장증상': False,
    # '연쇄구균생식소증상': False
}

tenaci = {
    '활주세균체표증상': False,
    # '활주세균입증상': False,
    '활주세균지느러미증상': False,
    '활주세균아가미증상': False
}

# entero = {
#     '여윔체표증상': False,
#     '여윔간증상': False
# }

miamien = {
    '스쿠티카체표증상': False,
    # '스쿠티카입증상': False,
    '스쿠티카지느러미증상': False,
    '스쿠티카아가미증상': False
}

vhsv = {
    '바이러스성체표증상': False,
    # '바이러스성복부증상': False,
    # '바이러스성항문증상': False,
    '바이러스성아가미증상': False,
    '바이러스성복수증상': False,
    '바이러스성간증상': False,
    # '바이러스성신장증상': False,
    # '바이러스성비장증상': False
}

# diseases = [edward, vibrio, strepto, tenaci, entero, miamien, vhsv]
diseases = [edward, vibrio, strepto, tenaci, miamien, vhsv]

def init_diseases():
    for disease in diseases:
        for key in disease.keys():
            disease[key] = False
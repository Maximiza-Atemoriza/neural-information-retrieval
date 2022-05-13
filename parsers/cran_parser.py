def parse_cran(path_to_file=None):
    if path_to_file == None:
        path_to_file = "datasets/Cran/cran.all.1400"
    documents = []
    with open(path_to_file, "r") as file:
        line = True
        document = {}
        info_type = ""
        while line:
            line = file.readline()
            if ".I" in line:
                documents.append(document)
                document = {}
                info_type = ".I"
                document[info_type] = line[3:-1]
            elif ".A" in line:
                info_type = ".A"
                document[info_type] = ''
            elif ".B" in line:
                info_type = ".B"
                document[info_type] = ''
            elif ".W" in line:
                info_type = ".W"
                document[info_type] = ''
            elif ".T" in line:
                info_type = ".T"
                document[info_type] = ''
            else:
                # remove end-of-line and add space 
                document[info_type] += line[:-1] + " "
        documents.append(document)
    return documents[1:]

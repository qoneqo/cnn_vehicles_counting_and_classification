import xlsxwriter

def save_to_xlsx(vehicles, nama_kendaraan, filename):
    workbook = xlsxwriter.Workbook('output_xlsx/'+filename+'.xlsx')
    worksheet = workbook.add_worksheet()
    
    char_key = ord('A')
    nama_kendaraan.append('waktu')
    for i in range(len(nama_kendaraan)):
        char = chr(char_key + i)
        worksheet.write(char+'1', nama_kendaraan[i])

    for i in range(len(vehicles)):
        num = i+2
        char_key = ord('A')
        for j in range(len(nama_kendaraan)):
            char = chr(char_key + j)
            worksheet.write(char+str(num), str(vehicles[i][j]))

    workbook.close()
import xlsxwriter

def save_to_xlsx(vehicles, nama_kendaraan, filename, folder_output):
    workbook = xlsxwriter.Workbook(folder_output+'/'+filename+'.xlsx')
    worksheet = workbook.add_worksheet()
    char_key = ord('A')
    nm_kendaraan = nama_kendaraan.copy()
    nm_kendaraan.append('tanggal')
    nm_kendaraan.append('waktu')
    
    for i in range(len(nm_kendaraan)):
        char = chr(char_key + i)
        worksheet.write(char+'1', nm_kendaraan[i])

    for i in range(len(vehicles)):
        num = i+2
        char_key = ord('A')
        for j in range(len(nm_kendaraan)):
            char = chr(char_key + j)
            worksheet.write(char+str(num), str(vehicles[i][j]))

    workbook.close()
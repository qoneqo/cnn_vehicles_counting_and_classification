import xlsxwriter

def save_to_xlsx(vehicles):
    workbook = xlsxwriter.Workbook('output_xlsx/total_vehicles.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'Sepeda Motor')
    worksheet.write('B1', 'Sepeda')
    worksheet.write('C1', 'Mobil Penumpang')

    worksheet.write('A2', str(vehicles[0]))
    worksheet.write('B2', str(vehicles[1]))
    worksheet.write('C2', str(vehicles[2]))

    workbook.close()
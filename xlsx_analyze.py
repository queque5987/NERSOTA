import pandas as pd
import openpyxl

file_name = 'xlm_entities_to_analyse_recognized_as_NE_per_total.xlsx'
df = pd.read_excel(file_name, engine='openpyxl')
org_t = df['OriginalToken'].values
npt = df['NE/Total'].values
ne = df['NE'].values
total = df['Total'].values
morethan20 = []
print(total)
for i, _ in enumerate(df['OriginalToken']):
    if total[i] < 20:
        continue
    morethan20.append([org_t[i], npt[i], ne[i], total[i]])

df = pd.DataFrame(morethan20)
df.to_excel("xlm_entities_to_analyse_recognized_as_NE_per_total_cut20below.xlsx", sheet_name= 'new_name', index= False, header= False)
# with pd.ExcelWriter('xlm_c_event.xlsx') as writer:
#     inventors[inventors.name == 'Nikola Tesla'].to_excel(writer, sheet_name='Nikola Tesla')
#     inventors[inventors.name == 'Thomas Edison'].to_excel(writer, sheet_name='Thomas Edison')
#     inventors[inventors.name == 'Henry Ford'].to_excel(writer, sheet_name='Henry Ford')   
# NERtesting
트위그팜 SOL 프로젝트 구어체 NER 인식률 향상

## toCSV.py
데이터셋 Tag 개수/종류 일치 메소드 사용 방법

```Python
def find_overlap_token(dir : Path, do_concat = False, do_drop = False, name = "", concat_tag_dict = {
     'PERSON' : ['PERSON', 'PS'],
     ...
     'TERM' : ['TM','TMI', 'TMIG', 'TMM']
    }, drop_tag_dict = {
    'PERSON' : ['PERSON', 'PS'],
    'ORG' : ['OGG', 'ORG'],
    'PRODUCT' : ['AFW', 'PRODUCT'],
    'WORK_OF_ART' : ['AFA', 'WORK_OF_ART']
    }):
```
```Python
# new_corpus_no_overlap -> 중복 제거된 데이터셋
dir = Path('new_corpus_no_overlap.csv의 위치')
do_drop = True
drop_tag_dict = {
  '합치고자 하는 태그' : ['합쳐질 태그', '합쳐질 태그', ... ], # 기본 세팅의 경우 PERSON, PS_... 태그가 PERSON으로 통합됨
  ... # '합쳐질 태그'에 기재되지 않은 태그는 삭제됨(아래 new_corpus_no_overlap.csv에 포함된 태그들 참조)
  ... # 삭제된 태그만 있는(drop 이후 ner.text가 ko_original과 같은) 열 또한 삭제됨
}
find_overlap_token(dir, do_drop = do_drop, drop_tag_dict)
```
new_corpus_no_overlap.csv에 포함된 태그들 ('-' 이후, 세부 태그의 경우는 _이하 절삭됨 ex) PS_PERSON -> PS, PERSON과 PS_... 태그가 동시에 있음)
```Python
[PERSON - 'PERSON' == 'PS',
     STUDY FIELD - 'FD', 
     THEORY - 'TR',
     ARTIFACTS - 'AF', ('AFA' == 'WORK_OF_ART'), ('AFW' == 'PRODUCT'),
     ORGANIZATION - 'OGG' == 'ORG',
     CIVILIZATION - 'CV',
     LOCATION - 'LC','LCG', 'LCP',
     DATE - 'DT',
     TIME - 'TI',
     QUANTITY - 'QT',
     EVENT - 'EV',
     ANIMAL - 'AM',
     PLANT - 'PT',
     MATERIAL - 'MT'
     TERM - 'TM','TMI', 'TMIG', 'TMM']
```

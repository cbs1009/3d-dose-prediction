# 3D Dose Prediction Utilities

## CT/RT 폴더의 Patient Name 확인

DICOM CT/RT 폴더 안의 파일에서 `PatientName`, `PatientID`, `Modality` 메타데이터를 확인하려면 `scripts/list_patient_names.py`를 사용하세요.

```bash
python scripts/list_patient_names.py /path/to/dataset
```

Windows 경로도 그대로 넣을 수 있습니다. 경로에 공백이 있으므로 따옴표로 감싸세요.

```powershell
python scripts/list_patient_names.py "C:\Users\M292670\Desktop\13_PTCOG_Oral_Presentation\4D CT Lung"
```

또는 PowerShell에서 이미 해당 폴더에 들어가 있다면 인자 없이 실행해도 됩니다.

```powershell
python .\list_patient_names.py
```

`pydicom`이 설치되어 있으면 더 다양한 DICOM 파일을 안정적으로 읽습니다. 실제 DICOM 파일에 PatientName 변경을 저장하려면 `pydicom`이 필요합니다.

```bash
python -m pip install -r requirements.txt
```

아래 세 가지 구조를 자동으로 인식합니다.

```text
/path/to/dataset/
  CT/
    patient_or_case_folder/*.dcm
  RT/
    patient_or_case_folder/*.dcm
```

```text
/path/to/dataset/
  case_001/
    CT/*.dcm
    RT/*.dcm
  case_002/
    CT/*.dcm
    RT/*.dcm
```

폴더명이 DICOM UID처럼 숫자와 점으로만 되어 있는 구조도 자동으로 인식합니다. 이 데이터셋 규칙에 맞춰 폴더명이 `1`로 시작하면 RT, `2`로 시작하면 CT로 분류합니다.

```text
/path/to/dataset/
  1.2.410.200113.1.20421.20260415003711893/  # RT
    *.dcm
  1.2.410.200113.1.22925.20260414201203811/  # RT
    *.dcm
  2.25.181506724508997602582525749623665176700/  # CT
    *.dcm
  2.25.187717231063694807607694361658911483649/  # CT
    *.dcm
```

각 환자/case 폴더 안에 UID 폴더들이 들어있는 구조도 됩니다. 예를 들어 `case_001/1.2...`는 RT, `case_001/2.25...`는 CT로 잡습니다. 이제 dataset 루트 아래를 재귀적으로 검색하므로 `4D CT Lung/101_HM10395/100_HM10395/1.2...`처럼 두 단계 이상 깊게 들어가 있어도 찾습니다. 즉, 21개 환자/case 폴더 각각에 UID 형태의 CT/RT series 폴더가 들어있는 구조라면 상위 dataset 루트만 지정하거나, 그 폴더에서 인자 없이 실행해도 자동으로 찾아 읽습니다.

CT와 RT 경로를 직접 지정할 수도 있습니다.

```bash
python scripts/list_patient_names.py --ct /path/to/CT --rt /path/to/RT
```

CSV로 저장하려면 `--csv`를 추가하세요.

```bash
python scripts/list_patient_names.py /path/to/dataset --csv patient_names.csv
```

CT/RT 폴더 바로 아래에 DICOM 파일이 있고 하위 폴더가 없는 경우에는 `--include-root-files`를 추가하세요. 단, `case_001/CT/*.dcm` 같은 case 폴더 구조에서는 자동으로 처리됩니다.

```bash
python scripts/list_patient_names.py /path/to/dataset --include-root-files
```

## PatientName을 직접 수정해서 바꾸기

가장 안전한 방법은 먼저 수정용 CSV 템플릿을 만든 다음, `new_patient_name` 열을 직접 편집하고 적용하는 것입니다.

1. 현재 이름을 확인하고 수정용 CSV를 만듭니다.

```bash
python scripts/list_patient_names.py /path/to/dataset --export-rename-csv rename_names.csv
```

Windows 예시는 아래와 같습니다.

```powershell
python scripts/list_patient_names.py "C:\Users\M292670\Desktop\13_PTCOG_Oral_Presentation\4D CT Lung" --export-rename-csv rename_names.csv
```

2. `rename_names.csv`를 Excel 등으로 열어서 `new_patient_name` 열에 바꿀 이름을 입력합니다. 바꾸지 않을 행은 빈칸으로 두세요.

```csv
current_patient_name,new_patient_name,groups,folders,dicom_files
OLD^NAME,NEW^NAME,case_001/CT; case_001/RT,/path/to/...,120
```

3. 먼저 dry run으로 몇 개 파일이 바뀔지 확인합니다. 이 단계에서는 실제 파일을 바꾸지 않습니다.

```bash
python scripts/list_patient_names.py /path/to/dataset --rename-csv rename_names.csv
```

4. 결과가 맞으면 `--apply`를 추가해서 실제 DICOM 파일에 저장합니다.

```bash
python scripts/list_patient_names.py /path/to/dataset --rename-csv rename_names.csv --apply
```

## PatientName을 명령어로 바로 바꾸기

간단히 한두 개 이름만 바꿀 때는 `--rename-patient OLD=NEW`를 사용할 수 있습니다. 기본값은 dry run입니다.

```bash
python scripts/list_patient_names.py /path/to/dataset --rename-patient 'OLD^NAME=NEW^NAME'
```

확인 후 실제 DICOM 파일에 저장하려면 `--apply`를 추가하세요.

```bash
python scripts/list_patient_names.py /path/to/dataset --rename-patient 'OLD^NAME=NEW^NAME' --apply
```

여러 이름을 한 번에 바꿀 수도 있습니다.

```bash
python scripts/list_patient_names.py /path/to/dataset \
  --rename-patient 'OLD^NAME=ANON001' \
  --rename-patient 'OTHER^NAME=ANON002' \
  --apply
```

터미널에서 하나씩 물어보게 하려면 `--interactive-rename`을 사용할 수도 있습니다.

```bash
python scripts/list_patient_names.py /path/to/dataset --interactive-rename --apply
```

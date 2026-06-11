import pandas as pd
import re

def parse_transcript(file_path):
    # 발화자 목록 정의 (추가 가능)
    speakers = ["주세형 교수님", "양수연 교수님", "이성준 교수님", "고건욱"]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    current_speaker = None
    current_content = []
    sequence = 1

    # 첫 발화자가 나오기 전까지의 메타데이터 스킵을 위한 플래그
    start_parsing = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 현재 줄이 발화자인지 확인
        if any(speaker == line for speaker in speakers):
            start_parsing = True
            # 이전 발화자의 내용을 리스트에 추가
            if current_speaker and current_content:
                data.append([sequence, current_speaker, " ".join(current_content)])
                sequence += 1
                current_content = []
            
            current_speaker = line
        else:
            # 발화자가 정해진 상태에서 텍스트가 나오면 내용으로 수집
            if start_parsing and current_speaker:
                # 하이퍼링크나 구분선 등 불필요한 데이터 제외 로직 (선택 사항)
                if not line.startswith("http") and not line.startswith("-"):
                    current_content.append(line)

    # 마지막 발화 내용 추가
    if current_speaker and current_content:
        data.append([sequence, current_speaker, " ".join(current_content)])

    return data

def save_to_excel(data, output_file):
    # 데이터프레임 생성
    df = pd.DataFrame(data, columns=['발화 순번', '발화자', '발화 내용'])
    
    # 엑셀 파일로 저장
    df.to_excel(output_file, index=False)
    print(f"성공: {output_file} 파일이 생성되었습니다.")

# 실행 영역
if __name__ == "__main__":
    input_filename = "이성준 교수님.md"  # 입력 파일명
    output_filename = "인터뷰_데이터_정리2.xlsx"  # 출력 파일명
    
    try:
        parsed_data = parse_transcript(input_filename)
        save_to_excel(parsed_data, output_filename)
    except FileNotFoundError:
        print(f"오류: '{input_filename}' 파일을 찾을 수 없습니다.")
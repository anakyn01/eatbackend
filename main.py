import os
import json
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# 로컬 개발용 .env 로드 (Render 환경에서는 Render 대시보드 Environment Variables 사용)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("warning: GROQ_API_KEY 환경변수를 찾을 수 없습니다!")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://heartbreakkid.co.kr",
        "https://heartbreakkid.co.kr",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    district: Optional[str] = "노원구"


class PlaceInfo(BaseModel):
    name: str
    category: str
    address: str
    lat: float
    lng: float
    is_registered: bool
    description: str


class ChatResponse(BaseModel):
    reply: str
    show_map: bool = False
    places: List[PlaceInfo] = []


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    user_message = request.message
    district = (request.district or "노원구").strip()

    system_prompt = (
        "당신은 박식하고 친절한 만능 AI 어시스턴트입니다. "
        f"사용자가 선택한 기본 기준 지역은 '{district}'입니다 (질문에 다른 지역이 명시되면 해당 지역을 우선 반영하세요). "
        "일상 대화, 코딩, 번역, 비즈니스, 맛집/학원 추천을 포함해 "
        "사용자가 묻는 모든 분야의 질문에 대해 정확하고 이해하기 쉽게 답변해 주세요. "
        "반드시 자연스럽고 매끄러운 한국어로 답변하세요.\n\n"
        "응답은 반드시 아래의 JSON 형식으로만 작성해야 합니다:\n"
        "{\n"
        '  "reply": "사용자에게 보여줄 친절한 답변 (마크다운 문법 사용 가능)",\n'
        '  "show_map": true 또는 false (맛집, 학원, 카페, 식당, 장소 추천 등 지도가 필요한 질문이면 true, 일반 대화면 false),\n'
        '  "places": [\n'
        "   {\n"
        '      "name": "업체명",\n'
        '      "category": "맛집 또는 학원 등 카테고리",\n'
        f'      "address": "서울 {district} 상세 주소",\n'
        '      "lat": 37.6551 (해당 지역의 대략적인 위도 숫자),\n'
        '      "lng": 127.0614 (해당 지역의 대략적인 경도 숫자),\n'
        '      "is_registered": true 또는 false (유명 프랜차이즈나 지도에 등록된 유명 업체면 true, 숨은 골목식당이나 가상의 추천/미등록 업체면 false),\n'
        '      "description": "대표 메뉴나 학원 특징 요약"\n'
        "   }\n"
        "  ]\n"
        "}\n"
        "주의: 일반 대화(show_map이 false일때)인 경우 places는 빈 배열 [] 로 보내세요"
    )

    show_map_flag = False
    places_list = []

    try:
        print(f"사용자 질문 수신 [{district}]: {user_message} -> Groq 초고속 응답 요청 중...")

        chat_completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.6,
            response_format={"type": "json_object"},
        )

        raw_content = chat_completion.choices[0].message.content
        print("Groq 응답 완료")

        parsed_data = json.loads(raw_content)
        ai_reply = parsed_data.get("reply", "답변을 생성하지 못했습니다")
        show_map_flag = bool(parsed_data.get("show_map", False))
        places_list = parsed_data.get("places", [])

    except Exception as e:
        print(f"Groq AI 호출중 에러 발생: {e}")
        ai_reply = "죄송합니다. 현재 AI 응답을 가져오는 중 문제가 발생했습니다. 서버 환경변수(GROQ_API_KEY)나 로그를 확인해 주세요."
        show_map_flag = False
        places_list = []

    return ChatResponse(
        reply=ai_reply,
        show_map=show_map_flag,
        places=places_list,
    )
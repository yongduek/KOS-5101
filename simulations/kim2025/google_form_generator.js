function createGoogleForm() {
  // 1. 설문지 기본 설정
  var form = FormApp.create('유학생 문화적응 및 자아존중감 설문조사');
  form.setDescription('본 설문은 학술 연구를 위한 목적으로 응답의 비밀이 보장되며, 수집된 모든 정보는 통계 처리용으로만 사용됩니다.');

  // 2. 기본 인적사항 섹션
  form.addSectionHeaderItem().setTitle('1. 기본 인적사항');
  form.addTextItem().setTitle('사용자아이디 (ID)').setRequired(true);
  form.addTextItem().setTitle('응답자의 국적').setRequired(true);
  form.addTextItem().setTitle('나이 (예: 25)').setRequired(true);

  form.addMultipleChoiceItem().setTitle('성별')
    .setChoiceValues(['남성', '여성']).setRequired(true);
  form.addMultipleChoiceItem().setTitle('과정')
    .setChoiceValues(['석사', '박사']).setRequired(true);
  form.addMultipleChoiceItem().setTitle('학기')
    .setChoiceValues(['1학기', '2학기', '3학기', '4학기', '5학기 이상']).setRequired(true);
  form.addMultipleChoiceItem().setTitle('한국어능력')
    .setChoiceValues(['상', '중', '하']).setRequired(true);
  form.addMultipleChoiceItem().setTitle('토픽(TOPIK) 레벨')
    .setChoiceValues(['4급', '5급', '6급', '원어민']).setRequired(true);
  form.addMultipleChoiceItem().setTitle('경제상황 (주관적 인식)')
    .setChoiceValues(['상', '중', '하']).setRequired(true);

  function addIndividualQuestions(targetQuestions, choices) {
    for (var i = 0; i < targetQuestions.length; i++) {
      form.addMultipleChoiceItem()
        .setTitle(targetQuestions[i])
        .setChoiceValues(choices)
        .setRequired(true);
    }
  }

  // 3. 자아존중감 검사 (RSES)
  form.addPageBreakItem().setTitle('2. 자아존중감 검사 (RSES)')
    .setHelpText('다음 질문을 읽고 평소 자신의 생각과 가장 잘 일치하는 항목을 선택해 주세요.');

  var rsesQ = [
    "1. 나는 다른 사람들과 마찬가지로 가치있는 사람이라고 생각한다. (I feel that I'm a person of worth, at least on an equal plane with others.)",
    "2. 나는 좋은 성품을 가졌다고 생각한다. (I feel that I have a number of good qualities.)",
    "3. 나는 대체로 실패한 사람이라고 생각한다. (All in all, I am inclined to feel that I am a failure.)",
    "4. 나는 대부분의 다른 사람들과 같이 일을 잘 할 수 있다. (I am able to do things as well as most other people.)",
    "5. 나는 자랑할 만한 것이 별로 없다. (I feel I do not have much to be proud of.)",
    "6. 나는 내 자신에 대해 긍정적인 태도를 가지고 있다. (I take a positive attitude toward myself.)",
    "7. 나는 내 자신에 대해 대체로 만족한다. (On the whole, I am satisfied with myself.)",
    "8. 나는 내 자신을 좀 더 존중할 수 있었으면 좋겠다. (I wish I could have more respect for myself.)",
    "9. 나는 가끔 내 자신이 쓸모없는 사람이라는 느낌이 든다. (I certainly feel useless at times.)",
    "10. 나는 때때로 내가 좋지 않은 사람이라고 생각한다. (At times I think I am no good at all.)"
  ];
  var rsesChoices = ["1. 전혀 그렇지 않다 (Strongly disagree)", "2. 그렇지 않은 편이다 (Disagree)", "3. 그런 편이다 (Agree)", "4. 매우 그렇다 (Strongly agree)"];
  addIndividualQuestions(rsesQ, rsesChoices);

  // 4. 문화적응스트레스 검사 (RSSIS/ASSIS)
  form.addPageBreakItem().setTitle('3. 문화적응스트레스 검사 (ASSIS)')
    .setHelpText('타국에서 생활하며 경험하실 수 있는 여러 스트레스 요인(향수병, 차별감, 두려움 등)에 대한 질문입니다.');

  var rssisQ = [
    "1. 나는 내 문화적 가치를 존중받지 못한다고 느낀다. (Others don't appreciate my cultural values.)",
    "2. 나는 내가 정당하게 받아야 할 대우를 거부당한다고 느낀다. (I feel that I am denied what I deserve.)",
    "3. 내 국적(출신국) 사람들 전체가 차별을 받는다고 느낀다. (I feel that my people are discriminated against.)",
    "4. 사람들이 비언어적(눈빛, 태도 등)으로 나에게 적대감을 표현한다. (People show hatred toward me nonverbally.)",
    "5. 나는 사회적 상황이나 모임에서 다르게 취급받는다. (I am treated differently in social situations.)",
    "6. 이 사회에서 나의 지위가 낮다고 느껴진다. (I feel that my status in this society is low.)",
    "7. 나의 신체적/인종적 특성 때문에 불이익을 받는다. (I am treated differently because of my race/color.)",
    "8. 일부 사람들은 내가 외국인이라는 이유로 나와 엮이는 것을 꺼린다. (Some people do not associate with me because of my ethnicity.)",
    "9. 나는 깊은 향수병(고향에 대한 그리움)을 느낀다. (I feel homesick.)",
    "10. 내가 떠나오면서 남겨둔 고향 사람들을 생각하면 상실감이 든다. (I feel a sense of loss for the people I left behind.)",
    "11. 고향의 가족들이 자주 생각난다. (I often think about my family in my home country.)",
    "12. 고국의 특정 장소나 사람들이 그립다. (I miss the places and people in my home country.)",
    "13. 나는 한국에 간(온) 후에 스트레스를 많이 받는다. (I generally experience a lot of stress after coming to Korea.)", // Kate (2021) modification
    "14. 현지인들 중 일부가 나에게 적대적이다. (I am treated with hostility.)",
    "15. 사람들이 나를 외부인(이방인) 취급한다. (People look at me as an outsider.)",
    "16. 한국어로 의사소통할 때 불안하고 긴장된다. (I feel nervous to communicate in Korean.)",
    "17. 이곳에 있는 것이 심리적으로 안전하지 않고 불안하다. (I feel insecure here.)",
    "18. 사람들 앞에서 문법적인 실수를 할까 봐 두렵다. (I am afraid of making mistakes in public.)",
    "19. 사람들이 나를 피하는 것 같다고 느낀다. (I feel that people avoid me.)",
    "20. 일부 현지인들의 태도 때문에 위축된다. (I am intimidated by some of the locals.)",
    "21. 이곳 생활 방식의 많은 변화들 때문에 극도로 피곤함을 느낀다. (Multiple changes in my life make me stressful.)",
    "22. 내가 가진 가치관과 현지의 가치관 간의 차이로 혼란스럽다. (I am overwhelmed by the differences in cultural values.)",
    "23. 한국의 음식이나 물리적 환경(날씨 등)에 적응하기 힘들다. (I find it difficult to adapt to local physical environments or food.)",
    "24. 한국의 기관 행정이나 절차를 다루는 것이 매우 스트레스다. (I have difficulty dealing with local administrative procedures.)",
    "25. 가족들을 남겨두고 유학을 온 것에 대해 죄책감을 느낀다. (I feel guilty leaving my family behind.)",
    "26. 내 가족들은 못 누리는데 나만 여기서 누리고 있다고 생각될 때 미안하다. (I feel guilty enjoying things here that my family can't.)",
    "27. 학업 진행이나 논문 관련 문제로 큰 우려가 있다. (I am worried about my academic progress.)",
    "28. 유학 중 재정적 지원이나 돈 문제가 충분하지 않아 걱정이다. (I lack adequate financial support.)",
    "29. 현지 사람들의 무례하거나 냉담한 시선에 자주 상처를 받는다. (I am hurt by the cold stares of some locals.)",
    "30. 이곳에서 나의 미래나 진로 전망에 대해 우려한다. (I am concerned about my future prospects here.)",
    "31. 나는 사회적 자원이나 지인(네트워크)이 부족하다. (I lack a good social network here.)",
    "32. 나의 문화를 무시하거나 부정적으로 얘기하는 것을 들을 때가 있다. (I hear negative comments about my culture.)",
    "33. 나는 외롭고 소외감을 느낀다. (I feel lonely and isolated.)",
    "34. 내 상황을 혼자 감당해야 한다는 생각에 슬퍼진다. (I feel sad when I think of my current situation.)",
    "35. 아파도 제대로 치료를 받지 못할까 봐 걱정된다. (I worry about getting proper healthcare if I get sick.)",
    "36. 한국인 동료나 학과 사람들과 교류할 때 마음이 편하지 않다. (I feel uneasy when interacting with locals in my department.)"
  ];

  var rssisChoices = ["1. 전혀 그렇지 않다 (Strongly disagree)", "2. 별로 그렇지 않다 (Disagree)", "3. 보통이다 (Neutral)", "4. 어느정도 그렇다 (Agree)", "5. 매우 그렇다 (Strongly agree)"];
  addIndividualQuestions(rssisQ, rssisChoices);

  // 5. 상호문화감수성 검사 (ISS)
  form.addPageBreakItem().setTitle('4. 상호문화감수성 검사 (ISS)')
    .setHelpText('타 문화권 사람들과 상호작용하는 여러분의 태도와 감정에 대한 질문입니다.');

  var issQ = [
    "1. 나는 다른 문화권 사람들과 상호작용하는 것을 즐긴다. (I enjoy interacting with people from different cultures.)",
    "2. 나는 타 문화권 사람들이 속이 좁다고 생각한다. (I think people from other cultures are narrow-minded.)",
    "3. 나는 타 문화권 사람들과 상호작용할 때 꽤 자신이 있다. (I am pretty sure of myself in interacting with people from different cultures.)",
    "4. 나는 타 문화권 사람들 앞에서 말하는 것이 매우 어렵다고 느낀다. (I find it very hard to talk in front of people from different cultures.)",
    "5. 나는 타 문화권 사람들과 상호작용할 때 무슨 말을 해야 할지 항상 알고 있다. (I always know what to say when interacting with people from different cultures.)",
    "6. 나는 타 문화권 사람들과 상호작용할 때 내가 원하는 만큼 사교적일 수 있다. (I can be as sociable as I want to be when interacting with people from different cultures.)",
    "7. 나는 타 문화권 사람들과 함께 있는 것을 좋아하지 않는다. (I don't like to be with people from different cultures.)",
    "8. 나는 타 문화권 사람들의 가치관을 존중한다. (I respect the values of people from different cultures.)",
    "9. 나는 타 문화권 사람들과 교류할 때 쉽게 화가 나고 짜증이 난다. (I get upset easily when interacting with people from different cultures.)",
    "10. 나는 타 문화권 사람들과 상호작용할 때 자신감을 느낀다. (I feel confident when interacting with people from different cultures.)",
    "11. 나는 문화적으로 다른 상대방에 대한 인상을 형성하기 전에 판단을 보류하는 편이다. (I tend to wait before forming an impression of culturally-distinct counterparts.)",
    "12. 나는 타 문화권 사람들과 함께 있을 때 종종 위축되거나 낙담한다. (I often get discouraged when I am with people from different cultures.)",
    "13. 나는 타 문화권 사람들에게 개방적인 태도를 가지고 있다. (I am open-minded to people from different cultures.)",
    "14. 나는 타 문화권 사람들과 상호작용할 때 매우 관찰력이 있다. (I am very observant when interacting with people from different cultures.)",
    "15. 나는 타 문화권 사람들과 교류할 때 종종 내 자신이 쓸모없다고 느낀다. (I often feel useless when interacting with people from different cultures.)",
    "16. 나는 타 문화권 사람들의 행동 방식을 존중한다. (I respect the ways people from different cultures behave.)",
    "17. 나는 타 문화권 사람들과 교류할 때 가능한 한 많은 정보를 얻으려고 노력한다. (I try to obtain as much information as I can when interacting with people from different cultures.)",
    "18. 나는 타 문화권 사람들의 의견을 받아들이지 않으려 한다. (I would not accept the opinions of people from different cultures.)",
    "19. 나는 상호작용 중 문화가 다른 상대방의 미묘한 의미에 민감하게 반응한다. (I am sensitive to my culturally-distinct counterpart's subtle meanings during our interaction.)",
    "20. 나는 내 문화가 다른 문화보다 우월하다고 생각한다. (I think my culture is better than other cultures.)",
    "21. 나는 교류 중에 문화가 다른 상대방에게 종종 긍정적인 반응을 보인다. (I often give positive responses to my culturally different counterpart during our interaction.)",
    "22. 나는 문화적으로 다른 사람들과 대면해야 하는 상황을 회피한다. (I avoid those situations where I will have to deal with culturally-distinct persons.)",
    "23. 나는 언어적 또는 비언어적 단서를 통해 문화가 다른 상대방에게 나의 이해를 자주 보여준다. (I often show my culturally-distinct counterpart my understanding through verbal or nonverbal cues.)",
    "24. 나는 문화가 다른 상대방과 나 사이의 차이점에서 즐거움을 느낀다. (I have a feeling of enjoyment towards differences between my culturally-distinct counterpart and me.)"
  ];
  addIndividualQuestions(issQ, rssisChoices);

  // 6. 종료 후 로깅
  Logger.log('설문지 자동 생성 완료! 확인 링크: ' + form.getEditUrl());
}

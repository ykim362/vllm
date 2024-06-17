from vllm import LLM, SamplingParams

from huggingface_hub import login
login(token="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")

#prompts = [
#    "Hello, my name is",
#    "The president of the United States is",
#    "The capital of France is",
#    "The future of AI is",
#    "DeepSpeed is a",
#]
prompts = [
   """You are an NLP assistant whose purpose is to solve reading comprehension problems. You will be provided questions on a set of passages and you will need to provide the answer as it appears in the passage. The answer should be in the same language as the question and the passage.ดาวแคระขาวที่ถูกค้นพบเป็นดวงแรกอยู่ในระบบดาวสามดวงใน 40 Eridani ซึ่งประกอบไปด้วยดาวสว่างในแถบลำดับหลัก 40 Eridani A ซึ่งโคจรอยู่ใกล้กับระบบดาวคู่ซึ่งมีดาวแคระขาว 40 Eridani B และดาวแคระแดงในแถบลำดับหลัก 40 Eridani C ฟรีดดริค วิลเฮล์ม เฮอร์เชล ได้ค้นพบคู่ดาว 40 Eridani B/C ตั้งแต่วันที่ 31 มกราคม ค.ศ. 1783[9] ต่อมา Friedrich Georg Wilhelm Struve ได้เฝ้าสังเกตในปี 1825 และ Otto Wilhelm Struve เฝ้าสังเกตในปี 1815[10][11] ครั้นถึงปี ค.ศ. 1910 Henry Norris Russel, Edward Charles Pickering และ Williamina Fleming จึงได้ค้นพบว่า ทั้ง ๆ ที่มันเป็นดาวที่จางแสงมาก แต่ 40 Eridani B จัดเป็นดาวที่มี spectral type A หรือมีแสงสีขาว[4] ในปี 1939 Russell มองย้อนไปในการสำรวจ[3]

Q: ใครเป็นผู้ค้นพบ ดาวแคระขาว?

Referring to the passage above, the correct answer to the given question is: ฟรีดดริค วิลเฮล์ม เฮอร์เชล

พระราชประวัติของสมเด็จพระนเรศวรมหาราชส่วนใหญ่ได้จากพงศาวดารอยุธยา ซึ่งมักมีการจดบันทึกในพระราชกรณียกิจ ซึ่งส่วนใหญ่บันทึกถึงการทำสงครามกับอาณาจักรเพื่อนบ้านเป็นอันมาก จนมองข้ามเกี่ยวกับเจ้านายฝ่ายในหรือพระมเหสีของพระองค์ อย่างไรก็ตามได้มีการปรากฏพระนามของเจ้านายฝ่ายใน ในเอกสารของต่างชาติ 5 ฉบับด้วยกัน ซึ่งได้แก่ จดหมายเหตุสเปน (History of the Philippines and Other Kingdom) ของบาทหลวงมาร์เชโล เด ริบาเดเนย์รา (Marchelo de Ribadeneira, O.F.M), จดหมายเหตุวันวลิต, พงศาวดารละแวก, คำให้การขุนหลวงหาวัด และพงศาวดารพม่าฉบับหอแก้ว ซึ่งปรากฏพระนามพระมเหสี 3-4 พระองค์ โดยมีพระนามดังนี้[39]

Q: สมเด็จพระนเรศวรมหาราชมี พระอัครมเหสีกี่คน?

Referring to the passage above, the correct answer to the given question is: 3-4


มาสเตอร์เชฟไทยแลนด์ เป็นรายการเกมโชว์ทำอาหาร ออกอากาศครั้งแรกในเดือนมิถุนายน พ.ศ. 2560 ทางสถานีโทรทัศน์สีกองทัพบกช่อง 7 โดยเปิดรับสมัครผู้เข้าแข่งขันที่มีอายุ 18 ปีขึ้นไป จากทุกอาชีพและทุกภาคของประเทศไทย[1]
โดยมีผู้เข้าแข่งขันจากทั่วประเทศสามารถผ่านมาได้ 120 คนและได้มาทำการคัดเลือกให้เหลือ 38 คน ก่อนจะทำการคัดให้เหลือ 16 คนสุดท้ายด้วยทักษะในครัวขั้นพื้นฐาน

Q: มาสเตอร์เชฟไทยแลนด์ซีซั้นแรกออกอากาศเมื่อใด ?

Referring to the passage above, the correct answer to the given question is: มิถุนายน พ.ศ. 2560

ภาษากงกณี(อักษรเทวนาครี: कोंकणी ; อักษรกันนาดา:ಕೊಂಕಣಿ; อักษรมาลายาลัม:കൊംകണീ ; อักษรโรมัน: Konknni) เป็นภาษากลุ่มอินโด-อารยัน โดยมีคำศัพท์จากภาษาดราวิเดียนปนอยู่ด้วย ได้รับอิทธิพลจากภาษาอื่นหลายภาษา ทั้งภาษาโปรตุเกส ภาษากันนาดา ภาษามราฐี ภาษาเปอร์เซีย และภาษาอาหรับ มีผู้พูดประมาณ 7.6 ล้านคน [1] [2]แต่เดิมเชื่อกันว่าภาษานี้เป็นภาษาถิ่นของภาษามราฐี แต่หลักฐานที่พบภาษากอนกานีเกิดก่อนภาษามราฐีนานมาก จารึกภาษากอนกานีพบครั้งแรกเมื่อ พ.ศ. 1730 ขณะที่จารึกภาษามราฐีพบครั้งแรก เมื่อราว พ.ศ. 2100

Q: ภาษากงกณี ถูกค้นพบเมื่อใด?

Referring to the passage above, the correct answer to the given question is: พ.ศ. 2100

เวสต์มินสเตอร์แอบบีย์ (English: Westminster Abbey) เดิมเป็นแอบบีย์ แต่ปัจจุบันเป็นโบสถ์ในนิกายแองกลิคันที่ตั้งอยู่ทางตะวันตกของพระราชวังเวสต์มินสเตอร์ในนครเวสต์มินสเตอร์ กรุงลอนดอน ประเทศอังกฤษ สถาปัตยกรรมที่เห็นอยู่ในปัจจุบันเป็นแบบสถาปัตยกรรมกอทิกเป็นส่วนใหญ่นอกจากหอคอยที่เป็นสถาปัตยกรรมฟื้นฟูกอทิก เป็นสถานที่ประกอบพิธีราชาภิเษกและที่ฝังพระบรมศพพระมหากษัตริย์อังกฤษและพระศพพระบรมวงศานุวงศ์ ระหว่างปี ค.ศ. 1546 ถึง 1556 แอบบีย์ได้รับเลื่อนฐานะขึ้นเป็นอาสนวิหาร ต่อมาในรัชสมัยของสมเด็จพระราชินีนาถเอลิซาเบธที่ 1 แอบบีย์นี้ก็ได้รับแต่งตั้งให้เป็นพระอารามหลวง (Royal Peculiar)

Q: ชื่อเดิมของ เวสต์มินสเตอร์แอบบีย์ คืออะไร?

Referring to the passage above, the correct answer to the given question is: แอบบีย์

เกิดเมื่อวันที่ 18 กรกฎาคม ค.ศ.1871 ที่เมืองตูริน ซึ่งเป็นเมืองหลวงของแคว้นปีเอมอนเต (พีดมอนต์) ซึ่งเป็นเขตการปกครองหนึ่งใน 20 แคว้นของประเทศอิตาลี เป็นลูกชายของนักเคมี และช่างภาพมือสมัครเล่น เมื่อบัลลามีอายุได้เพียง 8 ปี พ่อของเขาได้เสียชีวิตลง ทำให้ครอบครัวตกอยู่ในสภาวะยากลำบาก ในปี ค.ศ.1883 เพื่อที่จะช่วยเหลือแม่ บัลลาจึงเริ่มทำงานโดยการเป็น lithographer (ช่างทำภาพพิมพ์หิน)  ดังนั้นความสนใจทางด้านศิลปะของเขานั้นได้ถูกกระตุ้นมาตั้งแต่วัยเยาว์ [1]  ในปี ค.ศ. 1891 เขาได้เรียนในระยะสั้นๆ ที่  และ  ในตูริน ได้จัดแสดงงานศิลปะเป็นครั้งแรกภายใต้การอุปถัมภ์ของสมาคมศิลปะในเมืองของเขา ต่อจากนั้นได้เข้าเรียนที่มหาวิทยาลัยตูริน 

Q: จาโกโม บัลลา เกิดเมื่อไหร่?

Referring to the passage above, the correct answer to the given question is: 18 กรกฎาคม ค.ศ.1871



จังหวัดเลย เป็นจังหวัดหนึ่งในภาคตะวันออกเฉียงเหนือตอนบนของประเทศไทย ตั้งอยู่ในแอ่งสกลนครและอยู่ในกลุ่มจังหวัดภาคตะวันออกเฉียงเหนือตอนบน 1 ห่างจากกรุงเทพมหานครประมาณ  540 กิโลเมตร มีสภาพภูมิประเทศที่งดงาม อากาศหนาวเย็น เป็นแหล่งเพาะปลูกไม้ดอกไม้ประดับที่สำคัญแห่งหนึ่งของประเทศ และยังเป็นเมืองท่องเที่ยวที่สำคัญอีกด้วย

Q: จังหวัดเลย อยู่ภาคใดของไทย ?

Referring to the passage above, the correct answer to the given question is: ตะวันออกเฉียงเหนือ

ราชสกุล เป็นนามสกุลสำหรับผู้สืบเชื้อสายมาจากตระกูลวงศ์ในพระมหากษัตริย์ เกิดขึ้นเมื่อปี พ.ศ. 2455 หลังจากพระบาทสมเด็จพระมงกุฎเกล้าเจ้าอยู่หัว ทรงตราพระราชบัญญัตินามสกุลขึ้น โดยบัญญัติไว้ว่าผู้ที่สืบเชื้อสายทางบิดามาจากบรรพบุรุษคนเดียวกันให้ใช้ "นามสกุล" เดียวกัน  ทำให้คนไทยทุกคนมีนามสกุลใช้และทำให้ทราบว่าใครเป็นพี่น้องหรือสืบเชื้อสายมาจากผู้ใด[1] โดยผู้สืบเชื้อสายจากราชสกุล เรียกว่า ราชนิกุล และสกุลอันสืบเนื่องมาจากกรมพระราชวังบวรสถานมงคล (วังหน้า) และ กรมพระราชวังบวรสถานภิมุข (วังหลัง) นั้น เรียกว่า บวรราชสกุล

Q: ราชสกุล คืออะไร ?

Referring to the passage above, the correct answer to the given question is: นามสกุลสำหรับผู้สืบเชื้อสายมาจากตระกูลวงศ์ในพระมหากษัตริย์

กุบไล ข่าน (English: Kublai Khan) หรือ ซื่อจูหวางตี้ หรือ ซีโจ๊วฮ่องเต้ เป็นข่านแห่งจักรวรรดิมองโกลและยังเป็นฮ่องเต้องค์แรกของราชวงศ์หยวน

Q: กุบไลข่านคือใคร?

Referring to the passage above, the correct answer to the given question is:"""
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)
llm = LLM(model="/model/phi3-moe-rc-3_0_8",
          tokenizer="/model/phi3-moe-rc-3_0_8",
#llm = LLM(model="/home/data/transformers_cache/moe-test-phi/204600_hf_bfloat16",
#          tokenizer="microsoft/custom_llama2",
        #   trust_remote_code=True,
          dtype="bfloat16", tensor_parallel_size=2, max_model_len=8192, max_num_batched_tokens=8192)
# llm = LLM(model="facebook/opt-125m")
#llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", tensor_parallel_size=2, max_model_len=200)
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
print (outputs)
# import pdb;pdb.set_trace()
for output in outputs:
    prompt = output.prompt
    out= output.outputs
    print (prompt)
    print (out)
    print ('-------------')
    
    

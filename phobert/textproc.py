# textproc.py
# -*- coding: utf-8 -*-
import re, unicodedata, string
from typing import Dict, List, Tuple

LABELS_5 = ["very_negative","negative","neutral","positive","very_positive"]

# ===== Emojis (đã mở rộng) =====
EMO_POS = ["🤩","🥰","😍","❤️","👍","😎","👌","✨","🔥","💯","❤","♥","💕","💖","💗","💓","💞","💘","💝","💟","😄","😁","😃","🙂","😊","😌","🤗","👏","🙌","⭐","🌟","⚡","🎉","🥳","🔝","🆒","🆗","✅"]
EMO_NEG = ["😱","😡","🤬","💩","👎","😤","😞","😭","😠","😖","😣","😫","😩","🥵","🥶","🤢","🤮","😷","⚠","❌","🆘","💔","🥲","😓","😔","😕"]


SLANG_NEG = {
    # gốc đang có
    "vcl", "vkl", "vl",

    # chửi thề/teencode ngắn
    "dm", "đm", "dmm", "đmm", "dcm", "đcm", "vđ", "vcđ", "vđời", "cl", "cmm", "cc", "cục súc",
    "wtf", "wt*", "shit", "fuck", "fk", "fml",

    # dạng không dấu/teencode của chửi thề
    "ditme", "ditmemay", "dit me", "dmmay", "dmsp", "dm shop", "dkm", "dkmm",

    # mắng chửi nhẹ nhưng tiêu cực
    "khốn nạn", "láo", "lươn", "lươn lẹo", "bố láo", "xàm", "xàm l", "xàm lol", "xạo", "xạo l",
    "xạo ke", "xạo chó", "lừa", "lừa đảo", "treo đầu dê bán thịt chó",

    # phàn nàn kiểu slang
    "toang", "phèn", "dởm", "lởm", "lởm khởm", "rác", "rác rưởi", "tệ vl", "tệ vcl", "kém vl",
    "bố đời", "hãm", "kệch cỡm", "tụt mood", "tụt cmn mood",

    # viết tắt hay gặp trong review
    "k nên mua", "ko nên mua", "k đáng", "ko đáng", "k mua", "ko mua", "kém chất lượng", "hàng đểu",
    "hàng fake", "fake l", "fakel", "loot", "scam",

    # bôi đen/che bằng ký tự
    "đ*t", "đ**", "đ* m*", "địt m*", "l*n", "l**", "c**t", "c*t", "cụ c*", "ngu vcl", "ngu vl",
    "ngu như bò", "óc chó", "óc lợn",

    # biến thể không dấu các cụm trên (một số phổ biến)
    "luon leo", "xam lol", "xao lol", "xao ke", "lua dao", "rac ruoi", "kem chat luong",
    "hang de u", "hang deul", "hang fake", "oc cho", "oc lon",
}


# ===== Lexicon patterns =====
# ===== Lexicon patterns (tích cực, mở rộng cho review TMĐT) =====
POS_PATTERNS = [
    # các mẫu sẵn có (gom nhóm / chỉnh nhẹ cho gọn)
    r"\bđỉnh(?:\s+của\s+chóp)?\b",
    r"\bmãi\s*đỉnh\b",
    r"\bcực\s*phẩm\b",
    r"\b(best\s+of\s+best|best\s+choice)\b",
    r"\bquá\s*ok(?:e+)?\b",
    r"\bok\s*phết\b",
    r"\bổn\s*áp\b",
    r"\brecommend\s*mạnh\b",
    r"\byêu\s+lắm\s+lun\b",
    r"\b10\s*/\s*10\b",
    r"\bperfect\b",
    r"\bkhông\s+chê\s+vào\s+đâu\s+được?\b",
    r"\bquá\s+yêu\b",
    r"\btuyệt\s*vời\b",

    # bổ sung phổ biến trong review TMĐT
    r"\brất\s*tốt\b",
    r"\brất\s*hài\s*lòng\b|\bhài\s*lòng\b",
    r"\bưng\s*ý\b|\brất\s*ưng\b",
    r"\bxịn(?:\s*s[òo])?\b",                       # xịn / xịn sò / xịn xò
    r"\bđáng\s*(tiền|mua|giá|đồng\s*tiền)\b",
    r"\bchất\s*lượng\s*(tốt|ổn|ok)\b",
    r"\bngon\s*bổ\s*rẻ\b",
    r"\bgiao\s*nhanh\b|\bgiao\s*hàng\s*nhanh\b",
    r"\bđóng\s*gói\s*(cẩn\s*thận|kĩ|kỹ)\b",
    r"\bđúng\s*(mô\s*tả|như\s*(mô\s*tả|hình))\b",
    r"\bshop\s*(nhiệt\s*tình|uy\s*tín)\b",
    r"\btư\s*vấn\s*nhiệt\s*tình\b|\bphục\s*vụ\s*tốt\b",
    r"\bbảo\s*hành\s*tốt\b|\bhỗ\s*trợ\s*nhanh\b",

    # tiếng Anh thường gặp trong review
    r"\bgood\b|\bvery\s*good\b|\bgreat\b|\bexcellent\b",
    r"\blove\s*it\b|\bhighly\s*recommend\b|\bmust\s*buy\b",
    r"\bworth\s*it\b",

    # điểm/sao & ok-variants
    r"\b5\s*/\s*5\b|\b9\s*/\s*10\b|\b4\.5\s*/\s*5\b",
    r"\bok(?:e+|ie|i)?\b",                         # ok, oke, okie, oki
    r"\bổn\b|\bổn\s*áp\b",

    # khen hình thức/chất lượng dùng nhiều
    r"\bđẹp\b|\brất\s*đẹp\b|\bsiêu\s*đẹp\b",
    r"\bxuất\s*sắc\b|\btuyệt\s*hảo\b",
    r"\bbền\b|\brất\s*bền\b",
    r"\bđúng\s*size\b|\bvừa\s*vặn\b",
    r"\bđúng\s*hẹn\b|\bgiao\s*đúng\s*hẹn\b",
]


# ===== Lexicon patterns (tiêu cực, mở rộng cho review TMĐT) =====
NEG_PATTERNS = [
    # Chất lượng kém / chê thẳng
    r"\bkém\s*chất\s*lượng\b",
    r"\bchất\s*lượng\s*tệ\b",
    r"\brất\s*tệ\b|\bquá\s*tệ\b|\btệ\s*hại\b",
    r"\brác\b|\brác\s*rưởi\b",
    r"\bdởm\b|\blởm\b|\blởm\s*khởm\b",
    r"\bphí\s*tiền\b|\bkhông\s*đáng\s*tiền\b|\bđắt\s*vô\s*lý\b",

    # Hỏng / lỗi / hư hại vật lý
    r"\bhỏng\b|\bbị\s*hỏng\b|\blỗi\b|\bbị\s*lỗi\b|\bdefect\b|\bbroken\b",
    r"\bvỡ\b|\bmóp\b|\bxước\b|\brò\s*rỉ\b|\bchảy\s*nước\b",

    # Không đúng mô tả / sai size-màu / giao nhầm
    r"\bkhông\s+như\s+m[ôo]\s*tả\b|\bkhông\s+đúng\s+m[ôo]\s*tả\b|\bkhông\s+giống\s*hình\b",
    r"\bsai\s*m[âa]u\b|\bsai\s*size\b|\bsai\s*hàng\b|\bgửi\s*nhầm\b",

    # Giao hàng / thiếu hàng / chậm trễ
    r"\bgiao\s*hàng\s*(chậm|lâu|trễ|delay|kẹt\s*mãi)\b",
    r"\bgiao\s*thiếu\b|\bthiếu\s*hàng\b|\bthiếu\s*phụ\s*kiện\b",

    # Đóng gói kém / tem-seal
    r"\bđóng\s*gói\s*(cẩu\s*thả|sơ\s*sài|ẩu)\b|\bg[óo]i\s*ẩu\b",
    r"\bseal\s*(rách|bóc|mất)\b|\btem\s*(rách|bị\s*xé|mất)\b",

    # Hàng giả / không chính hãng
    r"\bhàng\s*giả\b|\bhàng\s*nhái\b|\bfake\b|\bscam\b|\bkhông\s*chính\s*hãng\b",

    # Trải nghiệm / cảm xúc tiêu cực
    r"\bthất\s*vọng\b|\bquá\s*thất\s*vọng\b|\bkinh\s*dị\b|\bthảm\s*họa\b",
    r"\bbực\s*mình\b|\bức\s*xúc\b|\btức\b",

    # Dịch vụ / hỗ trợ kém
    r"\bthái\s*độ\s*(kém|tệ)\b|\bphục\s*vụ\s*tệ\b",
    r"\bkh[ôo]ng\s*trả\s*lời\b|\bch[âa]m\s*trả\s*lời\b|\bt[ưu]\s*vấn\s*tệ\b",
]

# ===== Teencode (chiều "chuẩn" -> các biến thể) để chuẩn hoá nghịch =====
TEENCODE_INV: Dict[str, List[str]] = {
    # ===== Phủ định / mức độ =====
    "không": ["ko","kh","k","khong","hong","hông","hok","hem","hêm","khg","khg.","hông có","hong co","kh dc","ko dc","k dc","kh đc","ko đc","k đc","ko thể","k thể","kh thể"],
    "chưa": ["chua","chz","chưaaa","chưa có","chua co","ch chưa","ch vẫn","chv","chua lam","ch lam"],
    "chẳng": ["chang","chả","cha","chả có","chả co","chang co","chả thây","cha thay"],
    "rất": ["rat","rấttt","vl","vcl","vvcl","cực","cực kỳ","cực kì","ck","max","siêu","vãi","vãi l","vãi lồn","vãi chưởng","vãi nồi","quá trời","siêu siêu","quá xịn"],
    "quá": ["qua","wá","qa","qá","quaaaa","vãi","vãi chưởng","siêu","max","vl","vcl"],
    "hơi": ["hoi","h","hơi bị","hơi hơi","hơi xíu","hơi xí"],
    "khá": ["kha","khá ổn","kha on","khà khà","tạm ổn","tạm on"],
    "ổn": ["on","ổn áp","ok","oke","okela","okla","oklah","okie","oki","okila","ok phết","ok phet","ổn p","on ap"],

    # ===== Khẳng định / phổ dụng =====
    "có": ["co","c","cóa","có á","y","yes","yep","yup","ok","oke","oki"],
    "được": ["dc","đc","dk","đk","ok","oke","okie","oki","oklah","okela","hợp lý","hợp lí","hợp lí phết","ổn áp","on ap"],
    "rồi": ["r","roi","rùi","r nè","r nha","r nhoé","r nhé","r nà"],
    "đúng": ["dung","chuẩn","chính xác","chuan","chuẩn bài","chuẩn cmnr","chuẩn k cần chỉnh","chuan k can chinh"],

    # ===== Đại từ / xưng hô ngắn =====
    "tôi": ["t","toi","tui","tớ","mềnh","em","mình"],
    "mình": ["mk","mik","m","minh","mính","bản thân mình","btm"],
    "bạn": ["b","bn","b ơi","bro","cậu","bạn ơi","thím","các bác"],

    # ===== Câu hỏi / trạng từ =====
    "gì": ["j","ji","cj","cái j","cai j","gì v","gì dz"],
    "tại sao": ["ts","vì sao","sao dz","sao z","sao vậy","sao v"],
    "như thế nào": ["ntn","như nào","sao ntn","s ntn","như tn"],
    "vậy": ["v","z","dz","vậ","v zj"],
    "bây giờ": ["bg","bh","hnay","bjo","bây h"],
    "vì": ["tại","tại vì","bởi vì","tai vi","boi vi"],

    # ===== Danh mục TMĐT =====
    "sản phẩm": ["sp","s/phẩm","spham","san pham","sản ph","mặt hàng","mh","items","item", "san ph","sản ph","san_ph","s.phẩm","s pham","sp ham","s ph"],
    "đơn hàng": ["đh","don hang","đơn","order","od","đặt hàng","dat hang","đặt đơn","dat don"],
    "khuyến mãi": ["km","sale","gg","flash sale","fs","đại hạ giá","đhg","sale khủng","sale to"],
    "giảm giá": ["gg","sale off","down giá","deal","deal sốc","gíam","giam gia","giãm gia"],
    "quảng cáo": ["qc","ads","pr","quãng cáo","quancao"],
    "bảo hành": ["bh","bao hanh","bh chính hãng","bh chh","warranty","war"],
    "chính hãng": ["chh","auth","authen","authentic","hàng chính hãnh","chính hảng"],
    "cửa hàng": ["shop","shoppee","shope","gian hàng","gian hang","store"],

    # ===== Giao hàng / vận chuyển =====
    "giao hàng": ["ship","gh","giao","giao lẹ","ship lẹ","giao liền","giao liền tay","giao nhanh","ship nhanh","giao cấp tốc","ship cap toc"],
    "giao nhanh": ["ship nhanh","giao cấp tốc","giao siêu nhanh","giao 1 nốt nhạc","giao ngay và luôn"],
    "giao chậm": ["ship chậm","giao lâu","ship lâu","delay","treo đơn","kẹt mãi","kẹt don","kẹt đơn","kẹt kho","tắc kho","vướng kho"],
    "đúng hẹn": ["đúng h","giao đúng hẹn","đúng lịch"],
    "trễ hẹn": ["chậm hẹn","giao trễ","giao muộn","trễ giờ","trễ lịch"],

    # ===== Đóng gói / seal =====
    "đóng gói": ["đg","dong goi","pack","package","đóng g","đóng gói kỹ","đg kỹ","đg kĩ","đóng gói kĩ","pack kỹ","pack kĩ","đóng g kĩ","đóng g ky"],
    "đóng gói cẩn thận": ["đg kĩ","pack kĩ","đóng g kĩ","đóng gói kỹ","pack kỹ","gói kĩ","gói kỹ","đóng kĩ","đóng kỹ"],
    "đóng gói sơ sài": ["đóng ẩu","gói ẩu","đóng g ẩu","g ẩu","đóng cẩu thả","đg ẩu","pack ẩu","pack ẩu tả"],
    "rách seal": ["seal rách","bị rách seal","mất seal","seal mất","seal bóc","bóc tem","tem rách","tem bị xé"],
    "hộp móp": ["hộp méo","hộp bẹp","móp méo","bẹp góc","móp góc","bẹp hộp"],

    # ===== Mô tả / đúng sai / hình ảnh =====
    "đúng mô tả": ["đúng như mô tả","đúng như hình","đúng hệt mô tả","chuẩn mô tả","chuẩn như hình"],
    "không như mô tả": ["không đúng mô tả","không giống hình","ko như mta","không như hình","không giống mô tả","không chuẩn mô tả"],
    "hình ảnh": ["h/a","hình","ảnh","hình chụp","ảnh chụp","pic","hình thật","ảnh thật"],

    # ===== Chất lượng / trải nghiệm =====
    "chất lượng": ["chất lg","chất lg.","clg","chat luong","chatluong","chất lượng sp","cl sp","quality","qlt"],
    "hài lòng": ["hailong","hai long","rất hài lòng","quá hài lòng","siêu hài lòng","ưng","ưng lắm","ưng phết","ưng ý","ưng ý phết"],
    "đáng tiền": ["đáng lắm","xứng đáng","đáng đồng tiền","đáng mua","đáng giá","đáng đồng tiền bát gạo"],
    "không đáng tiền": ["phí tiền","không đáng","không đáng mua","phí tiền oan","phí của"],
    "thất vọng": ["that vong","tv","tụt mood","siêu thất vọng","fail","disappointed","thất vọng tràn trề"],
    "tệ": ["te","tệ vl","tệ vcl","tệ quá","tệ thật sự","tệ hại"],
    "kém": ["kem","dởm","kém chất lượng","kém xịn","kém bền"],
    "bình thường": ["bt","btw","bth","binh thuong","thường thôi","tạm", "bth", "bình tg", "binh_thuong","b.t"],
    "xuất sắc": ["xuat sac","xs","xuất xắc","rất xuất sắc","quá xuất sắc"],
    "ngon bổ rẻ": ["nbr","ngon rẻ","ngon-bổ-rẻ","ngon bổ re"],

    # ===== Điểm số / đánh giá =====
    "5/5": ["5 / 5","5-5","5 sao","full sao","⭐⭐⭐⭐⭐","★★★★★","5*"],
    "4/5": ["4 / 5","4-5","4 sao","★★★★☆","4*"],
    "10/10": ["10 / 10","10-10","10đ","10d","10*","mười trên mười","10/ 10"],

    # ===== Thuộc tính sản phẩm =====
    "màu": ["mau","color","clr","màu sắc","ms"],
    "kích thước": ["size","kt","kthuoc","kích cỡ","kc","sz","cỡ"],
    "vừa vặn": ["fit","fit chuẩn","vừa y","vừa ôm","đúng size","chuẩn size"],
    "sai size": ["lệch size","size lệch","ship nhầm size","size sai","sai kích thước"],
    "chất liệu": ["clieu","chất liêu","material","fabric"],
    "độ bền": ["độ bền bỉ","bền bỉ","bền","durable","độ trâu","trâu bò"],

    # ===== Điện tử / hiệu năng =====
    "pin": ["battery","batt","pin trâu","trâu pin","pin ổn","pin on","pin ok"],
    "sạc": ["sac","chg","charge","sạc nhanh","sạc nhanh qc","sạc nhanh pd"],
    "âm thanh": ["am thanh","am_thanh","sound","audio","âm ổn","âm ok","bass","treble"],
    "màn hình": ["man hinh","man_hinh","mh","display","scr","screen","m.hinh","màn  hình"],
    "hiệu năng": ["hieu nang","perf","performance","mượt","mượt mà","mượt","lag","giật","lag giật","drop fps"],
    "nhiệt độ": ["nhiet do","nhiet_do","nhiệt","nóng","máy nóng","nong may","ấm máy"],
    "kết nối": ["ket noi","connect","cnct","bluetooth","bt","wifi","wi-fi","nfc","typec","type-c"],

    # ===== Thời trang / giày dép =====
    "form": ["phom","phôm","form dáng","form chuẩn","form ôm","form rộng"],
    "co giãn": ["co gian","co dãn","đàn hồi","độ giãn","giãn tốt"],
    "đường may": ["duong may","đ.may","may chỉ","đường chỉ","chỉ may"],
    "chất vải": ["chat vai","vải vóc","chất liệu vải","vải ok","vai ok"],
    "trơn tru": ["trơn tru","trơn chu","trơn tru lướt","mướt"],

    # ===== Mỹ phẩm / thực phẩm =====
    "mùi": ["mui","mùi hương","hương","h mùi","thơm","thom"],
    "hạn sử dụng": ["hsd","han sd","date","ngày hết hạn","ngay het han","expired","exp"],
    "đóng tuýp": ["tuyp","tuýp","tube","chai tuýp","dạng tuýp"],
    "kem chống nắng": ["kcn","chong nang","ccn","kem cn"],

    # ===== Dịch vụ / hỗ trợ =====
    "hỗ trợ": ["ho tro","ho_tro","support","sppt","htro","tư vấn","tu van","tv"],
    "phản hồi": ["phan hoi","phanhoi","feedback","fb","phản ứng","phản hồi nhanh"],
    "đổi trả": ["doi tra","đổi/trả","return","trả hàng","tra hang","refund","rf"],
    "bảo mật": ["bao mat","secure","security"],

    # ===== Thêm các cụm “dirtyfy” hay gặp =====
    "đóng gói kỹ": ["đg kỹ","pack kỹ","đóng kỹ","đóng kĩ","pack kĩ"],
    "giao liền": ["giao lẹ","ship lẹ","giao ngay","giao trong ngày","ship now","ship ngay"],
    "hàng chính hãng": ["hàng auth","hàng authentic","hàng chuẩn auth","chính hãnh","chính hảng"],
    "hàng fake": ["fake","hàng đểu","hang de u","hang deul","hàng nhái","super fake","s.fake"],

    # ===== Cụm tiêu cực/than phiền gom về chuẩn =====
    "giao hàng chậm": ["ship chậm","giao lâu","giao trễ","trễ hẹn","delay","kẹt mãi","kẹt đơn","kẹt kho","om hàng","om don"],
    "thiếu hàng": ["giao thiếu","thiếu phụ kiện","thiếu pk","thiếu đồ","thiếu chi tiết","thiếu linh kiện"],
    "đóng gói ẩu": ["đg ẩu","đóng ẩu tả","pack ẩu","gói ẩu","đóng sơ sài","đóng cẩu thả"],
    "không phản hồi": ["không trả lời","k trả lời","ko rep","kh rep","bỏ tin nhắn","seen không trả lời","seen k rep"],
    "thái độ kém": ["thái độ tệ","cọc cằn","trả lời cộc lốc","hách dịch","khó chịu","nhân viên tệ","nv tệ"],

    # ===== Bổ sung viết sai chính tả phổ biến =====
    "tuyệt vời": ["tuyet voi","tuyet vời","tuyệt vòi","tuyet vời","tuyetvoi"],
    "đẹp": ["dep","đẹpp","đẹppp","đepp","đẹpppp"],
    "xấu": ["xau","xấu òm","xấu tệ","xấu quắc"],
    "bền": ["ben","bền bỉ","bền lâu","rất bền","siêu bền"],
    "giá rẻ": ["gia re","giá rẽ","rẻ","rẻ bèo","rẻ phèn","rẻ hú hồn"],
    "đắt": ["dat","đắc","đắt đỏ","giá chát","chát quá"],

    # ===== Từ đồng nghĩa ngắn gọn thường gặp =====
    "uy tín": ["uytin","uý tín","tin cậy","đáng tin","đáng tin cậy"],
    "tốc độ": ["toc do","speed","spd","nhanh","nhah","nhahh","nhan"],
    "tiện lợi": ["tien loi","tiện dụng","tiện ích","tiện","tiện phết"],
    "dễ dùng": ["de dung","dễ xài","dễ sử dụng","dễ sd","de sd","easy to use"],
    "khó dùng": ["kho dung","khó xài","khó sd","khó sử dụng"],

    # ===== Một số biến thể “no dấu” có khoảng trắng / gạch dưới =====
    "âm thanh": ["am thanh","am_thanh"],
    "màn hình": ["man hinh","man_hinh"],
    "nhiệt độ": ["nhiet do","nhiet_do"],
    "dịch vụ": ["dich vu","dich_vu"],
    "hỗ trợ": ["ho tro","ho_tro"],
    "phản hồi": ["phan hoi","phan_hoi"],
    "chất lượng": ["chat luong","chat_luong"],
    "trải nghiệm": ["trai nghiem","trai_nghiem"],
    "cẩu thả": ["cau tha","cau_tha"],

    # ===== Một số câu khen/chê gọn gàng =====
    "rất tốt": ["tot vcl","tot vl","good lắm","good phết","quá tốt","tốt xịn","tốt tuyệt"],
    "quá tệ": ["te vl","te vcl","tệ vc","tệ dã man","tệ quá","tệ cực"],
    "tệ quá": ["te qua","tệ qa","te qa","tệ qá","te quá"],
    "rất đẹp": ["đẹp xỉu","đẹp xỉu up xỉu down","đẹp mlem","đẹp phết","đẹp lun","đep lun","đẹp mê ly"],
    "siêu nhanh": ["nhanh vãi","nhanh khủng","nhanh kinh khủng","siêu tốc","cực nhanh"],
}

# ===== Accent recover helpers =====
# ===== Accent recover helpers (ưu tiên ngữ cảnh TMĐT) =====
ACCENT_MAP = {
    # ==== Giữ nguyên & sửa cho phổ quát ====
    "may":"máy","nong":"nóng","mat":"mát","on":"ổn","kem":"kém","tot":"tốt","te":"tệ",
    "thatvong":"thất vọng","tuyetvoi":"tuyệt vời","tuyet":"tuyệt",
    "hai_long":"hài lòng","hai long":"hài lòng",
    "dang":"đáng","gia":"giá","dat":"đắt","re":"rẻ","ton":"tốn","tien":"tiền","mua":"mua",
    "pin":"pin","am":"âm","thanh":"thanh","am thanh":"âm thanh","am_thanh":"âm thanh",
    "dong":"đóng","goi":"gói","dong goi":"đóng gói","donggoi":"đóng gói",
    "nhanh":"nhanh","cham":"chậm","tre":"trễ","lau":"lâu","som":"sớm","ben":"bền","yeu":"yếu",
    "man":"màn","hinh":"hình","man hinh":"màn hình","man_hinh":"màn hình",
    "sac":"sạc","sac du phong":"sạc dự phòng","sacduphong":"sạc dự phòng",
    "nhiet":"nhiệt","do":"độ","nhiet do":"nhiệt độ","nhiet_do":"nhiệt độ",
    "dich vu":"dịch vụ","dichvu":"dịch vụ","ho tro":"hỗ trợ","ho_tro":"hỗ trợ",
    "phan hoi":"phản hồi","phanhoi":"phản hồi",
    "chat luong":"chất lượng","chatluong":"chất lượng",
    "trai nghiem":"trải nghiệm","trai_nghiem":"trải nghiệm",
    "cau tha":"cẩu thả","cau_tha":"cẩu thả",
    "dep":"đẹp","xau":"xấu","xuat sac":"xuất sắc","xuat_sac":"xuất sắc",
    "qua":"quá","rat":"rất","cuc":"cực","cuc ky":"cực kỳ","cuc_ky":"cực kỳ",
    "khong":"không","khong nen":"không nên",
    "dang tien":"đáng tiền","dang gia":"đáng giá",
    "giong":"giống","mo ta":"mô tả","mo_ta":"mô tả",
    "binh thuong":"bình thường","binh_thuong":"bình thường",
    "thich":"thích","rat thich":"rất thích","rat_thich":"rất thích","ung":"ưng",

    # ==== Sàn TMĐT: đơn hàng / vận chuyển ====
    "don hang":"đơn hàng","donhang":"đơn hàng","dat hang":"đặt hàng","dat don":"đặt đơn",
    "giao hang":"giao hàng","ship nhanh":"ship nhanh","ship cham":"ship chậm",
    "giao cham":"giao chậm","giao tre":"giao trễ","tre hen":"trễ hẹn","dung hen":"đúng hẹn",
    "treo don":"treo đơn","ket don":"kẹt đơn","ket kho":"kẹt kho","tac kho":"tắc kho",
    "nhan hang":"nhận hàng","nhan duoc hang":"nhận được hàng",

    # ==== Đóng gói / tình trạng hộp / tem-seal ====
    "dong goi ky":"đóng gói kỹ","dong goi ki":"đóng gói kĩ","goi ki":"gói kĩ","goi ky":"gói kỹ",
    "dong goi so sai":"đóng gói sơ sài","dong au":"đóng ẩu","goi au":"gói ẩu",
    "hop":"hộp","mop":"móp","meo":"méo","bep":"bẹp","rach":"rách","sun":"sún",
    "tem":"tem","seal":"seal","rach seal":"rách seal","mat seal":"mất seal","boc tem":"bóc tem",
    "tem rach":"tem rách","tem xe":"tem xé","tem bi xe":"tem bị xé",

    # ==== Sai/đúng mô tả, hình ảnh, thiếu hàng ====
    "dung mo ta":"đúng mô tả","dung nhu mo ta":"đúng như mô tả","dung nhu hinh":"đúng như hình",
    "khong nhu mo ta":"không như mô tả","khong giong hinh":"không giống hình",
    "thieu hang":"thiếu hàng","giao thieu":"giao thiếu","thieu phu kien":"thiếu phụ kiện",

    # ==== Size/màu / thuộc tính thời trang ====
    "size":"size","sai size":"sai size","dung size":"đúng size","vua van":"vừa vặn",
    "form":"form","phom":"phôm","co gian":"co giãn","dan hoi":"đàn hồi","chat vai":"chất vải",
    "duong may":"đường may","duong chi":"đường chỉ","chuan form":"chuẩn form",

    # ==== Chất lượng / cảm nhận chung ====
    "hai long":"hài lòng","rat hai long":"rất hài lòng","ung y":"ưng ý","ung y phet":"ưng ý phết",
    "dang mua":"đáng mua","dang dong tien":"đáng đồng tiền","khong dang tien":"không đáng tiền",
    "phi tien":"phí tiền","kem chat luong":"kém chất lượng","do gia":"đồ giả","hang de u":"hàng đểu",
    "xuat sac":"xuất sắc","tuyet hao":"tuyệt hảo","tot qua":"tốt quá","qua tot":"quá tốt",
    "te qua":"tệ quá","te hai":"tệ hại","that vong":"thất vọng","kinh di":"kinh dị","tham hoa":"thảm họa",

    # ==== Bảo hành / đổi trả / hỗ trợ ====
    "bao hanh":"bảo hành","doi tra":"đổi trả","hoan tien":"hoàn tiền",
    "doi hang":"đổi hàng","tra hang":"trả hàng","khieu nai":"khiếu nại",
    "tu van":"tư vấn","phan hoi nhanh":"phản hồi nhanh","ho tro nhanh":"hỗ trợ nhanh",
    "khong tra loi":"không trả lời","cham tra loi":"chậm trả lời","thai do kem":"thái độ kém",

    # ==== Điện tử / hiệu năng / kết nối ====
    "sac nhanh":"sạc nhanh","sac nhanh qc":"sạc nhanh QC","sac nhanh pd":"sạc nhanh PD",
    "bluetooth":"bluetooth","wifi":"wifi","wi fi":"wi-fi","nfc":"nfc","type c":"type-c","typec":"type-c",
    "hieu nang":"hiệu năng","muot":"mượt","lag":"lag","giat":"giật","drop fps":"drop fps",
    "loa":"loa","micro":"micro","am bass":"âm bass","am treble":"âm treble",

    # ==== Màn hình / hiển thị ====
    "do sang":"độ sáng","do tuong phan":"độ tương phản","goc nhin":"góc nhìn",
    "do phan giai":"độ phân giải","man hinh dep":"màn hình đẹp","lech mau":"lệch màu",

    # ==== Mỹ phẩm / thực phẩm / tiêu dùng ====
    "mui":"mùi","mui huong":"mùi hương","thom":"thơm","hang su dung":"hạn sử dụng",
    "hsd":"hsd","date":"date","ngay het han":"ngày hết hạn","het han":"hết hạn",
    "kem chong nang":"kem chống nắng","kcn":"kcn","duong am":"dưỡng ẩm","cap am":"cấp ẩm",
    "sua tam":"sữa tắm","dau goi":"dầu gội","dau xa":"dầu xả","sua rua mat":"sữa rửa mặt",

    # ==== Giá / khuyến mãi / thanh toán ====
    "gia re":"giá rẻ","gia chat":"giá chát","giam gia":"giảm giá","khuyen mai":"khuyến mãi",
    "uu dai":"ưu đãi","ma giam gia":"mã giảm giá","voucher":"voucher",
    "thanh toan":"thanh toán","cod":"cod","vi dien tu":"ví điện tử",

    # ==== Màu sắc (thường gặp trong review) ====
    "mau den":"màu đen","mau trang":"màu trắng","mau do":"màu đỏ","mau xanh":"màu xanh",
    "mau nau":"màu nâu","mau be":"màu be","mau kem":"màu kem","mau ghi":"màu ghi",
    "mau hong":"màu hồng","mau tim":"màu tím","mau vang":"màu vàng",

    # ==== Từ hình thái sai/không dấu phổ biến ====
    "bẹp":"bẹp","méo":"méo","mop meo":"móp méo","rach nat":"rách nát",
    "chu an toan":"chưa an toàn","kem ben":"kém bền","ben bi":"bền bỉ",

    # ==== Cụm khen/chê ngắn gọn (không dấu/thiếu dấu) ====
    "rat tot":"rất tốt","qua te":"quá tệ","rat dep":"rất đẹp","sieu dep":"siêu đẹp",
    "rat ben":"rất bền","sieu ben":"siêu bền","ngon bo re":"ngon bổ rẻ","dang tien":"đáng tiền",

    # ==== Biến thể có gạch dưới/space ====
    "chat_luong":"chất lượng","trai_nghiem":"trải nghiệm","am_thanh":"âm thanh",
    "man_hinh":"màn hình","nhiet_do":"nhiệt độ","dich_vu":"dịch vụ","ho_tro":"hỗ trợ",
    "phan_hoi":"phản hồi","binh_thuong":"bình thường","xuat_sac":"xuất sắc",

    # Neutral shorthand
    "bt": "bình thường",
    "bth": "bình thường",
    "b.t": "bình thường",
    "binh thg": "bình thường",
    "binh tg": "bình thường",

    # Product shorthands / broken forms
    "san ph": "sản phẩm",
    "san_ph": "sản phẩm",
    "s pham": "sản phẩm",
    "s ph": "sản phẩm",
    "s.pham": "sản phẩm",
    "san phan": "sản phẩm",      # typo → phẩm

    # Negative shorthands
    "te qua": "tệ quá",
    "te qa": "tệ quá",
    "tệ qa": "tệ quá",
}


# ===== Bigram hints (ưu tiên khôi phục dấu theo ngữ cảnh TMĐT) =====
BIGRAM_HINTS = {
    # Chất lượng / cảm nhận
    ("rất","tốt"): 2.2, ("rất","đẹp"): 2.0, ("rất","bền"): 2.0,
    ("rất","hài"): 1.8,  # → hài lòng
    ("rất","thích"): 1.6,
    ("rất","xứng"): 1.6,  # → xứng đáng
    ("rất","đáng"): 1.8,  # → đáng tiền/đáng mua
    ("rất","tệ"): 2.0, ("quá","tệ"): 2.1, ("quá","tuyệt"): 1.8,  # tuyệt vời
    ("cực","kỳ"): 1.8, ("cực","kì"): 1.8,
    ("siêu","đẹp"): 1.8, ("siêu","bền"): 1.8, ("siêu","nhanh"): 1.8,
    ("chất","lượng"): 2.2, ("kém","chất"): 2.1, ("xuất","sắc"): 2.0,
    ("ngon","bổ"): 2.0, ("bổ","rẻ"): 2.0,

    # Giá trị / giá cả
    ("đáng","tiền"): 2.2, ("đáng","mua"): 1.8, ("đáng","giá"): 1.8,
    ("phí","tiền"): 2.1, ("giá","rẻ"): 1.9, ("giá","chát"): 1.8,
    ("không","đáng"): 1.9, ("đắt","vô"): 1.6,  # → vô lý

    # Giao hàng / thời gian
    ("giao","nhanh"): 2.2, ("giao","chậm"): 2.2, ("giao","trễ"): 2.1,
    ("đúng","hẹn"): 2.0, ("trễ","hẹn"): 2.0,
    ("kẹt","đơn"): 1.8, ("kẹt","kho"): 1.8, ("tắc","kho"): 1.8,
    ("nhận","hàng"): 1.6,

    # Đóng gói / tình trạng hộp / tem-seal
    ("đóng","gói"): 2.2, ("gói","kĩ"): 2.0, ("gói","kỹ"): 2.0,
    ("đóng","ẩu"): 2.1, ("sơ","sài"): 2.0, ("cẩu","thả"): 2.0,
    ("rách","seal"): 2.1, ("mất","seal"): 2.0, ("bóc","tem"): 1.9, ("tem","rách"): 2.0,
    ("hộp","móp"): 2.0, ("hộp","méo"): 2.0, ("bẹp","góc"): 1.8,

    # Đúng/sai mô tả – hình ảnh
    ("đúng","mô"): 2.1,  # → mô tả
    ("đúng","như"): 1.9,  # → như mô tả/hình
    ("không","như"): 2.1, ("không","giống"): 2.0,
    ("giống","hình"): 1.9, ("mô","tả"): 2.1,

    # Size / màu
    ("đúng","size"): 2.0, ("sai","size"): 2.1, ("lệch","size"): 1.8,
    ("vừa","vặn"): 2.0, ("màu","sắc"): 1.8, ("lệch","màu"): 1.7,

    # Dịch vụ / hỗ trợ
    ("phản","hồi"): 1.9, ("hỗ","trợ"): 2.0, ("tư","vấn"): 1.8,
    ("thái","độ"): 2.0, ("thái","độ"): 2.0, ("dịch","vụ"): 1.9,
    ("không","trả"): 1.8, ("chậm","trả"): 1.7,  # → trả lời
    ("uy","tín"): 1.7,

    # Điện tử / hiệu năng / kết nối
    ("màn","hình"): 2.2, ("âm","thanh"): 2.2, ("pin","trâu"): 2.0,
    ("sạc","nhanh"): 2.0, ("máy","nóng"): 2.2, ("nhiệt","độ"): 2.0,
    ("kết","nối"): 1.8, ("wifi","yếu"): 1.8, ("bluetooth","yếu"): 1.6,
    ("độ","sáng"): 1.7, ("tương","phản"): 1.6,  # → độ tương phản

    # Thời trang / chất liệu
    ("form","chuẩn"): 1.7, ("chất","vải"): 1.9, ("đường","may"): 2.0,
    ("co","giãn"): 1.8, ("đàn","hồi"): 1.7, ("vải","đẹp"): 1.6,

    # Mỹ phẩm / thực phẩm
    ("mùi","hương"): 2.0, ("thơm","nhẹ"): 1.6, ("hạn","sử"): 1.9,  # → sử dụng
    ("hạn","dùng"): 1.9, ("hết","hạn"): 2.0, ("date","xa"): 1.6,

    # Đổi trả / hoàn tiền
    ("đổi","trả"): 2.1, ("hoàn","tiền"): 2.1, ("trả","hàng"): 1.9,
    ("khiếu","nại"): 1.8, ("xử","lý"): 1.7,

    # Hàng giả / vi phạm
    ("hàng","giả"): 2.1, ("hàng","nhái"): 2.1, ("hàng","đểu"): 2.0,
    ("không","chính"): 1.9,  # → chính hãng
    ("treo","đầu"): 1.7, ("bán","thịt"): 1.7,  # → dê / chó (ngữ cố định)

    # Thiếu/nhầm hàng
    ("thiếu","hàng"): 2.0, ("giao","thiếu"): 2.0, ("thiếu","phụ"): 1.9,  # → phụ kiện
    ("gửi","nhầm"): 1.8, ("sai","hàng"): 1.9,

    # Các cặp phổ biến khác
    ("đáng","mua"): 1.8, ("đáng","giá"): 1.7, ("đẹp","xuất"): 1.6,  # → xuất sắc
    ("bền","bỉ"): 1.9, ("vỡ","góc"): 1.7, ("xước","nhẹ"): 1.6, ("bong","tróc"): 1.7,

    ("sản", "phẩm"): 2.5,
    ("san", "pham"): 2.5,        # phòng khi prev token chưa có dấu
    ("tệ", "quá"): 2.2,
    ("te", "qua"): 2.2,
    ("bình", "thường"): 2.0,
    ("binh", "thuong"): 2.0,
}


# ---------- Tiền biên dịch regex TEENCODE_INV ----------
# Tạo danh sách (pattern, replacement) đã biên dịch để chạy nhanh
_TEENCODE_PATTERNS: List[Tuple[re.Pattern, str]] = []
def _compile_teencode_patterns():
    global _TEENCODE_PATTERNS
    if _TEENCODE_PATTERNS:
        return
    pairs = []
    for canonical, variants in TEENCODE_INV.items():
        for v in variants:
            # dùng \b nếu variant là 1 token, nếu có khoảng trắng thì match "thô"
            if " " in v:
                pat = re.compile(re.escape(v), flags=re.IGNORECASE)
            else:
                pat = re.compile(rf"\b{re.escape(v)}\b", flags=re.IGNORECASE)
            pairs.append((pat, canonical))
    # sắp xếp để ưu tiên variant dài hơn (tránh "ok" ăn mất "oklah")
    pairs.sort(key=lambda x: len(x[0].pattern), reverse=True)
    _TEENCODE_PATTERNS = pairs

# ===== Core normalize (DÙNG CHUNG CHO TRAIN & INFER) =====
def normalize_text(s: str) -> str:
    _compile_teencode_patterns()
    s = unicodedata.normalize("NFC", str(s).strip())
    # emoji → token
    for e in EMO_POS: s = s.replace(e, " EMO_POS ")
    for e in EMO_NEG: s = s.replace(e, " EMO_NEG ")
    s = s.lower()

    # Chuẩn hoá theo TEENCODE_INV (map nghịch)
    for pat, rep in _TEENCODE_PATTERNS:
        s = pat.sub(rep, s)

    # thay thế cụm thường gặp (bổ sung một số bạn đã dùng)
    repl = {
        "vl":"rất","okeee":"ok","ưng":"rất thích","siêu siêu":"rất",
        "siêu thất vọng":"rất thất vọng","mãi đỉnh":"rất tốt",
        "best of best":"rất tốt","best choice":"rất tốt","đỉnh của chóp":"rất tốt",
    }
    for k,v in repl.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s, flags=re.IGNORECASE)

    # co khoảng trắng
    return re.sub(r"\s+"," ", s).strip()

def count_lexicon(s: str):
    txt = s.lower()
    pos = sum(1 for p in POS_PATTERNS if re.search(p, txt))
    neg = sum(1 for p in NEG_PATTERNS if re.search(p, txt))
    pos += txt.count("emo_pos"); neg += txt.count("emo_neg")
    return pos, neg

def     sentiment_prefix(s: str, max_tag=1):
    pos, neg = count_lexicon(s)
    pos = min(pos, max_tag); neg = min(neg, max_tag)
    prefix = []
    if pos>0: prefix.append(f"__POS{pos}__")
    if neg>0: prefix.append(f"__NEG{neg}__")
    return (" ".join(prefix) + " " + s) if prefix else s

def maybe_segment(text, use_seg=False):
    if not use_seg: return text
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        return text

# ===== Diacritics helpers =====
def approx_diacritic_ratio(s: str) -> float:
    vowels_with_tone = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    s_low = str(s).lower()
    cnt_diac = sum(ch in vowels_with_tone for ch in s_low)
    cnt_letters = sum(ch.isalpha() for ch in s_low)
    return (cnt_diac / max(1, cnt_letters))

def strip_accents_simple(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = s.replace("đ","d").replace("Đ","D")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def _choose_variant(token_lc: str, prev_lc: str = None):
    cand = ACCENT_MAP.get(token_lc)
    if cand and prev_lc:
        score = 1.0 + BIGRAM_HINTS.get((prev_lc, cand.split()[0]), 0.0)
        return cand, score
    if cand:
        return cand, 1.0
    return None, 0.0

def restore_diacritics(text: str) -> str:
    tokens = text.split(" ")
    out_tokens, prev_norm = [], None
    for tok in tokens:
        prefix, suffix, core = "", "", tok
        while core and core[0] in string.punctuation:
            prefix += core[0]; core = core[1:]
        while core and core[-1] in string.punctuation:
            suffix = core[-1] + suffix; core = core[:-1]
        if not core:
            out_tokens.append(prefix + suffix); prev_norm = None; continue
        base = strip_accents_simple(core.lower())
        best, _ = _choose_variant(base, prev_norm)
        replaced = best if best else core
        if core.isupper():
            replaced = replaced.upper()
        elif core[0].isupper():
            replaced = replaced[0].upper() + replaced[1:]
        out_tokens.append(prefix + replaced + suffix)
        prev_norm = replaced.split()[0].lower()
    return " ".join(out_tokens)

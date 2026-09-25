# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 26

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:

1. Xe ở làn giữa phía dưới (gần camera, rọi đèn pha rất sáng xuống mặt đường): AI rất dễ bị nhiễu bởi vệt sáng đèn pha phản chiếu trên mặt đường bê tông, dẫn đến việc vẽ bounding box bị kéo dài quá mức bao trọn cả quầng sáng thay vì chỉ ôm sát thân xe theo đúng quy tắc.
2. Các xe ở xa phía làn bên phải đang di chuyển ra xa (chỉ lộ cụm đèn hậu màu đỏ, thân xe tối màu chìm vào hậu cảnh ban đêm): AI rất dễ bỏ sót hoàn toàn (False Negative) do độ tương phản kém và thiếu viền thân xe rõ rệt.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.

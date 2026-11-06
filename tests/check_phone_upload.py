"""手机照片上传的针对性检查；不训练、不修改正式图片。"""
import io
from pathlib import Path
import sys
import tempfile

from PIL import Image
from werkzeug.datastructures import FileStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as service


def encoded(size, fmt='JPEG', orientation=None):
    image = Image.new('RGB', size, (90, 140, 180))
    stream = io.BytesIO()
    options = {}
    if orientation:
        exif = image.getexif()
        exif[274] = orientation
        options['exif'] = exif
    image.save(stream, format=fmt, **options)
    stream.seek(0)
    return stream


def main():
    # 常见 1200 万像素照片：先按拍摄方向转正，再缩小。
    pixels, original = service.read_image(FileStorage(encoded((4032, 3024), orientation=6), filename='phone.jpg'))
    assert original == (3024, 4032)
    assert pixels.shape[:2] == (2309, 1732)
    assert pixels.shape[0] * pixels.shape[1] <= service.MAX_PIXELS
    # 真正的 HEIF 编码文件，不是改后缀的 JPEG。
    pixels, original = service.read_image(FileStorage(encoded((4032, 3024), 'HEIF'), filename='phone.heic'))
    assert original == (4032, 3024) and pixels.shape[:2] == (1732, 2309)
    client = service.app.test_client()
    previous = service.OUTPUT
    with tempfile.TemporaryDirectory(prefix='codex_phone_upload_') as directory:
        service.OUTPUT = Path(directory)
        try:
            # sigma=0 只检查大图上传、保存和尺寸元数据，不做耗时质量实验。
            result = client.post('/api/denoise', data={'image': (encoded((4032, 3024)), 'phone.jpg'), 'model': 'bm3d', 'sigma': '0'})
            assert result.status_code == 200, result.json
            data = result.json
            assert data['resized'] and data['original_width'] == 4032
            assert (data['width'], data['height']) == (2309, 1732)
            with Image.open(Path(directory) / data['id'] / 'output.png') as image:
                assert image.size == (2309, 1732)
            # 小 HEIC 实际调用普通页面默认模型，验证格式与已有模型可衔接。
            result = client.post('/api/denoise', data={'image': (encoded((73, 65), 'HEIF'), 'phone.heic')})
            assert result.status_code == 200, result.json
            assert not result.json['resized'] and result.json['model'] == service.DEFAULT_MODEL
            # 缩小后尺寸可能接近，但原始尺寸不匹配仍必须拒绝。
            result = client.post('/api/denoise', data={'image': (encoded((4032, 3024)), 'phone.jpg'), 'reference': (encoded((4000, 3000)), 'gt.jpg')})
            assert result.status_code == 400, result.json
            result = client.post('/api/denoise', data={'image': (io.BytesIO(b'not heic'), 'bad.heic')})
            assert result.status_code == 400
        finally:
            service.OUTPUT = previous
    print('PASS: phone JPEG resize, orientation, HEIC decode, default inference, result size metadata, mismatched reference, corrupt HEIC')


if __name__ == '__main__':
    main()

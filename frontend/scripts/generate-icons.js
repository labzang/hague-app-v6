// 간단한 아이콘 생성 스크립트
// 실제 프로덕션에서는 디자이너가 만든 아이콘을 사용하세요

const fs = require('fs');
const path = require('path');

// SVG 아이콘 생성 함수
function generateSVGIcon(size) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg">
  <rect width="${size}" height="${size}" fill="#3b82f6"/>
  <text x="50%" y="50%" font-family="Arial, sans-serif" font-size="${size * 0.4}" font-weight="bold" fill="white" text-anchor="middle" dominant-baseline="middle">RAG</text>
</svg>`;
}

// public 디렉토리 생성
const publicDir = path.join(__dirname, '../public');
if (!fs.existsSync(publicDir)) {
  fs.mkdirSync(publicDir, { recursive: true });
}

// SVG 아이콘 생성 (실제로는 PNG가 필요하지만, SVG로 시작)
const icon192 = generateSVGIcon(192);
const icon512 = generateSVGIcon(512);

fs.writeFileSync(path.join(publicDir, 'icon-192x192.svg'), icon192);
fs.writeFileSync(path.join(publicDir, 'icon-512x512.svg'), icon512);

console.log('아이콘 파일이 생성되었습니다.');
console.log('참고: 실제 프로덕션에서는 PNG 형식의 아이콘을 사용하세요.');
console.log('온라인 도구를 사용하여 SVG를 PNG로 변환할 수 있습니다:');
console.log('- https://cloudconvert.com/svg-to-png');
console.log('- 또는 디자이너에게 192x192, 512x512 PNG 아이콘을 요청하세요.');


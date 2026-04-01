/**
 * Mock data for CL-GRU4Rec+RP Demo
 * Product catalog for rental/e-commerce demo
 */

export interface Product {
  id: string;
  name: string;
  category: string;
  price: number;
  image: string;
  description: string;
}

export interface User {
  id: string;
  email: string;
  name: string;
}

export const CATEGORIES = [
  "Điện",
  "Chiếu sáng",
  "Công tắc/Ổ điện",
  "Công cụ",
  "Thiết bị thông minh",
] as const;

export const PRODUCTS: Product[] = [
  // Chiếu sáng
  { id: "p1", name: "Đèn LED Philips 9W", category: "Chiếu sáng", price: 85000, image: "💡", description: "Đèn LED tiết kiệm điện, tuổi thọ cao" },
  { id: "p2", name: "Bóng đèn LED 5W", category: "Chiếu sáng", price: 45000, image: "💡", description: "Bóng đèn LED nhỏ gọn, ánh sáng trắng" },
  { id: "p3", name: "Đèn downlight âm trần", category: "Chiếu sáng", price: 120000, image: "💡", description: "Đèn âm trần trang trí, ánh sáng dịu nhẹ" },
  { id: "p4", name: "Đèn đường LED 30W", category: "Chiếu sáng", price: 350000, image: "💡", description: "Đèn đường ngoài trời, chống nước" },

  // Ổ điện & Công tắc
  { id: "p5", name: "Ổ điện thông minh", category: "Công tắc/Ổ điện", price: 180000, image: "🔌", description: "Ổ điện WiFi, điều khiển từ xa" },
  { id: "p6", name: "Công tắc cảm ứng", category: "Công tắc/Ổ điện", price: 95000, image: "🔌", description: "Công tắc cảm ứng tự động" },
  { id: "p7", name: "Ổ cắm 3 chân", category: "Công tắc/Ổ điện", price: 35000, image: "🔌", description: "Ổ cắm thông dụng 3 lỗ" },
  { id: "p8", name: "Aptomat 1P 2P", category: "Công tắc/Ổ điện", price: 65000, image: "⚡", description: "Aptomat chống giật, an toàn" },

  // Thiết bị thông minh
  { id: "p9", name: "Remote đa năng", category: "Thiết bị thông minh", price: 150000, image: "📱", description: "Remote điều khiển TV, máy lạnh" },
  { id: "p10", name: "Cảm biến chuyển động", category: "Thiết bị thông minh", price: 220000, image: "📡", description: "Cảm biến PIR, phát hiện chuyển động" },
  { id: "p11", name: "Bộ chia HDMI 4K", category: "Thiết bị thông minh", price: 280000, image: "📺", description: "HUB chia HDMI 4 cổng 4K" },

  // Công cụ
  { id: "p12", name: "Kìm điện đa năng", category: "Công cụ", price: 145000, image: "🔧", description: "Kìm cắt, bứt dây điện" },
  { id: "p13", name: "Bồ đo điện tử", category: "Công cụ", price: 380000, image: "📐", description: "Đo điện áp, dòng điện, trở kháng" },

  // Phụ kiện điện
  { id: "p14", name: "Dây điện 2×1.5", category: "Điện", price: 12000, image: "🔗", description: "Dây dẫn điện cuộn 100m" },
  { id: "p15", name: "Bộ tản nhiệt LED", category: "Điện", price: 45000, image: "🌡️", description: "Tản nhiệt cho LED power" },
];

// Sample sessions for demo
export const SAMPLE_SESSIONS = [
  { id: "s1", products: ["p1", "p5", "p2"], name: "Session: Mua đèn + ổ điện" },
  { id: "s2", products: ["p6", "p10", "p9"], name: "Session: Nhà thông minh" },
  { id: "s3", products: ["p12", "p14", "p7"], name: "Session: Thi công điện" },
];

export function getProductById(id: string): Product | undefined {
  return PRODUCTS.find(p => p.id === id);
}

export function getProductsByCategory(category: string): Product[] {
  return PRODUCTS.filter(p => p.category === category);
}

export function getRandomProducts(count: number): Product[] {
  const shuffled = [...PRODUCTS].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
}

export function getPopularProducts(count: number = 6): Product[] {
  return PRODUCTS.slice(0, count);
}

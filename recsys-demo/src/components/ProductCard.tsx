/**
 * Product Card Component
 * Displays product with click handler
 * Works with real API Product type
 */

"use client";

import { Product } from "@/lib/api-client";

interface ProductCardProps {
  product: Product;
  onClick?: () => void;
  variant?: "default" | "compact";
}

// Get emoji icon based on category
function getCategoryIcon(category: string): string {
  const categoryIcons: Record<string, string> = {
    "Коконы для новорожденных": "👶",
    "Электрокачели": "🎠",
    "Качели, шезлонги": "🪑",
    "Коляски": "👼",
    "Манеж": "🏠",
    "Ходунки": "🚶",
    "Стульчики для кормления": "🍼",
    default: "📦",
  };
  return categoryIcons[category] || categoryIcons.default;
}

export function ProductCard({
  product,
  onClick,
  variant = "default",
}: ProductCardProps) {
  const displayPrice = product.price > 0
    ? `${product.price.toLocaleString('ru-RU')} ₽/ngày`
    : `${product.price_sell?.toLocaleString('ru-RU') || 'N/A'} ₽`;

  if (variant === "compact") {
    return (
      <button
        onClick={onClick}
        className="w-full text-left p-2 rounded hover:bg-gray-100 transition-colors flex items-center gap-2"
      >
        <span className="text-2xl">{getCategoryIcon(product.main_category)}</span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate text-gray-900">{product.name}</p>
          <p className="text-xs text-gray-500">{product.main_category}</p>
        </div>
        <p className="text-sm font-semibold text-blue-600">
          {displayPrice}
        </p>
      </button>
    );
  }

  return (
    <button
      onClick={onClick}
      className="w-full bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow p-4 border border-gray-200 hover:border-blue-300 text-left"
    >
      <div className="aspect-square bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg flex items-center justify-center mb-3">
        <span className="text-6xl">{getCategoryIcon(product.main_category)}</span>
      </div>
      <h3 className="font-semibold text-gray-900 mb-1 line-clamp-2 min-h-[2.5rem]">
        {product.name}
      </h3>
      <p className="text-sm text-gray-500 mb-1">{product.brand || ''}</p>
      <p className="text-xs text-gray-400 mb-2 truncate">{product.main_category}</p>
      <p className="text-lg font-bold text-blue-600">
        {displayPrice}
      </p>
    </button>
  );
}

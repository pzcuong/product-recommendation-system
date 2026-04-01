/**
 * Recommendation Panel Component
 * Displays CL-GRU4Rec+RP recommendations with component scores
 */

"use client";

import { Product, getProductById } from "@/lib/mock-data";
import { RecommendationResult } from "@/lib/recommendation-engine";
import { ProductCard } from "./ProductCard";

interface RecommendationPanelProps {
  result: RecommendationResult | null;
  onProductClick?: (productId: string) => void;
}

export function RecommendationPanel({ result, onProductClick }: RecommendationPanelProps) {
  if (!result) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold mb-4">🎯 Gợi ý cho bạn</h2>
        <p className="text-gray-500 text-center py-8">
          Chưa có gợi ý. Hãy xem sản phẩm để nhận gợi ý!
        </p>
      </div>
    );
  }

  const { recommendations, confidence } = result;

  // Get confidence level
  const getConfidenceLevel = (conf: number) => {
    if (conf < 0.4) return { text: "THẤP", color: "text-yellow-600", bg: "bg-yellow-50" };
    if (conf < 0.7) return { text: "TRUNG BÌNH", color: "text-blue-600", bg: "bg-blue-50" };
    return { text: "CAO", color: "text-green-600", bg: "bg-green-50" };
  };

  const confLevel = getConfidenceLevel(confidence);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">🎯 Gợi ý cho bạn</h2>
        <div className={`px-3 py-1 rounded-full text-xs font-semibold ${confLevel.bg} ${confLevel.color}`}>
          Độ tin cậy: {confLevel.text} ({(confidence * 100).toFixed(0)}%)
        </div>
      </div>

      {/* Component weights visualization */}
      <div className="mb-4 p-3 bg-gray-50 rounded-lg text-sm">
        <p className="font-semibold text-gray-700 mb-2">Trọng số Components:</p>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="text-center">
            <p className="font-medium text-blue-600">GRU4Rec</p>
            <p className="text-gray-500">Sequential</p>
          </div>
          <div className="text-center">
            <p className="font-medium text-purple-600">Contrastive</p>
            <p className="text-gray-500">Similarity</p>
          </div>
          <div className="text-center">
            <p className="font-medium text-green-600">Re-Purchase</p>
            <p className="text-gray-500">History</p>
          </div>
        </div>
      </div>

      {/* Recommendations grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {recommendations.map((rec, idx) => {
          const product = getProductById(rec.productId);
          if (!product) return null;

          return (
            <div key={rec.productId} className="relative">
              <div className="absolute top-2 left-2 bg-blue-600 text-white text-xs font-bold px-2 py-1 rounded-full z-10">
                #{idx + 1}
              </div>
              <ProductCard product={product} onClick={() => onProductClick?.(rec.productId)} />
              <div className="mt-2 p-2 bg-gray-50 rounded text-xs space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-600">GRU:</span>
                  <span className="font-medium text-blue-600">{rec.gruScore.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">CL:</span>
                  <span className="font-medium text-purple-600">{rec.clScore.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">RP:</span>
                  <span className="font-medium text-green-600">{rec.rpScore.toFixed(3)}</span>
                </div>
                <div className="flex justify-between border-t pt-1 mt-1">
                  <span className="font-semibold">Total:</span>
                  <span className="font-bold text-gray-900">{rec.finalScore.toFixed(3)}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

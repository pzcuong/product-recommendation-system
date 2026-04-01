/**
 * CL-GRU4Rec+RP Recommendation Engine (Demo)
 * Simulates the 3 components: GRU4Rec, Contrastive Learning, Re-Purchase
 */

"use client";

import { Product, getProductById } from "./mock-data";

export interface RecommendationScore {
  productId: string;
  finalScore: number;
  gruScore: number; // Sequential pattern
  clScore: number; // Contrastive similarity
  rpScore: number; // Re-purchase frequency
}

export interface RecommendationResult {
  recommendations: RecommendationScore[];
  sessionProducts: Product[];
  confidence: number; // GRU confidence based on session length
}

/**
 * Simulated GRU4Rec scoring based on sequential patterns
 */
function calculateGRUScore(
  productId: string,
  sessionIds: string[],
  allProducts: Product[],
): number {
  // Simulate sequential pattern scoring
  // Products in same category get higher scores
  const product = getProductById(productId);
  if (!product) return 0;

  let score = Math.random() * 0.3; // Base random score

  // Category co-occurrence boost
  const sessionCategories = new Set(
    sessionIds.map((id) => getProductById(id)?.category).filter(Boolean),
  );

  if (sessionCategories.has(product.category)) {
    score += 0.4; // Same category boost
  }

  // Position-based boost (recency)
  const lastProducts = sessionIds.slice(-3);
  if (lastProducts.length > 0) {
    score += 0.2 * (lastProducts.length / 3);
  }

  return Math.min(score, 0.95);
}

/**
 * Simulated Contrastive Learning similarity score
 */
function calculateCLScore(productId: string, sessionIds: string[]): number {
  // Simulate embedding similarity
  // Products with similar IDs (simulating similar embeddings) get higher scores
  const productNum = parseInt(productId.slice(1));
  const sessionNums = sessionIds.map((id) => parseInt(id.slice(1)));

  let similarity = 0;
  for (const sessionNum of sessionNums) {
    // Higher similarity for close product numbers (simulating embedding proximity)
    const diff = Math.abs(productNum - sessionNum);
    if (diff <= 2) similarity += 0.3;
    else if (diff <= 5) similarity += 0.1;
  }

  return Math.min(similarity, 0.8);
}

/**
 * Re-Purchase score based on frequency in session
 */
function calculateRPScore(productId: string, sessionIds: string[]): number {
  const count = sessionIds.filter((id) => id === productId).length;
  if (count === 0) return 0;

  // Normalize by session length
  return count / sessionIds.length;
}

/**
 * Generate recommendations using CL-GRU4Rec+RP fusion
 */
export function generateRecommendations(
  sessionIds: string[],
  allProducts: Product[],
  k: number = 6,
): RecommendationResult {
  // Calculate GRU confidence based on session length
  const sessionLength = sessionIds.length;
  const confidence = Math.min(0.3 + (sessionLength / 10) * 0.5, 0.95);

  // Adaptive weights based on session length
  const gruWeight = 0.5 + Math.min(sessionLength / 20, 0.3); // 0.5 - 0.8
  const clWeight = 0.25 - Math.min(sessionLength / 50, 0.1); // 0.15 - 0.25
  const rpWeight = 0.25; // Fixed

  // Calculate scores for all products not in session
  const sessionSet = new Set(sessionIds);
  const scores: RecommendationScore[] = [];

  for (const product of allProducts) {
    if (sessionSet.has(product.id)) continue;

    const gruScore = calculateGRUScore(product.id, sessionIds, allProducts);
    const clScore = calculateCLScore(product.id, sessionIds);
    const rpScore = calculateRPScore(product.id, sessionIds);

    const finalScore =
      gruWeight * gruScore + clWeight * clScore + rpWeight * rpScore;

    scores.push({
      productId: product.id,
      finalScore,
      gruScore,
      clScore,
      rpScore,
    });
  }

  // Sort by final score and take top-k
  scores.sort((a, b) => b.finalScore - a.finalScore);
  const topScores = scores.slice(0, k);

  // Get product details
  const sessionProducts = sessionIds
    .map((id) => getProductById(id))
    .filter(Boolean) as Product[];

  return {
    recommendations: topScores,
    sessionProducts,
    confidence,
  };
}

/**
 * Get fallback recommendations for cold start
 */
export function getColdStartRecommendations(
  allProducts: Product[],
  k: number = 6,
): RecommendationResult {
  const recommendations: RecommendationScore[] = allProducts
    .slice(0, k)
    .map((p) => ({
      productId: p.id,
      finalScore: Math.random() * 0.5 + 0.3,
      gruScore: Math.random() * 0.3,
      clScore: Math.random() * 0.2,
      rpScore: 0,
    }));

  return {
    recommendations,
    sessionProducts: [],
    confidence: 0.2, // Low confidence for cold start
  };
}

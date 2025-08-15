# Synthetic Datasets Analysis for GraphMamba Testing

**Date**: January 2025  
**Purpose**: Analyze available synthetic datasets to identify suitable candidates for GraphMamba testing

---

## 1. Dataset Categories Overview

### 1.1 Contagion Models
These datasets simulate information/influence spreading processes:

#### **synthetic_icm_ba** ✅ **RECOMMENDED**
- **Model**: Independent Cascade Model on Barabási-Albert network
- **Structure**: 79 edges, 29 timestamps, 80 nodes
- **Randomness**: **LOW** - Deterministic cascade patterns
- **Edge Labels**: Binary (1=exists, 0=doesn't exist)
- **Characteristics**: 
  - Clear temporal progression
  - Deterministic spreading rules
  - Good for learning temporal patterns
- **Status**: Already tested with GraphMamba (79.09% AP)

#### **synthetic_ltm_ba** ✅ **RECOMMENDED**
- **Model**: Linear Threshold Model on Barabási-Albert network
- **Structure**: 288 edges, multiple timestamps, ~150 nodes
- **Randomness**: **LOW** - Threshold-based deterministic activation
- **Edge Labels**: Binary (1=exists, 0=doesn't exist)
- **Characteristics**:
  - Threshold-based activation (more predictable than ICM)
  - Clear temporal structure
  - Good for learning influence patterns
- **Status**: **READY FOR TESTING**

#### **synthetic_cc_ws** ⚠️ **MODERATE RANDOMNESS**
- **Model**: Community-based Contagion on Watts-Strogatz network
- **Structure**: 50 edges, 10 timestamps, ~40 nodes
- **Randomness**: **MEDIUM** - Some random community formation
- **Edge Labels**: Binary (1=exists, 0=doesn't exist)
- **Characteristics**:
  - Community-based spreading
  - Some temporal structure
  - Moderate predictability
- **Status**: **READY FOR TESTING**

#### **synthetic_sd_ba** ❌ **HIGH RANDOMNESS**
- **Model**: Stochastic Diffusion on Barabási-Albert network
- **Structure**: 23 edges, continuous timestamps, ~30 nodes
- **Randomness**: **HIGH** - Continuous random diffusion
- **Edge Labels**: Binary (0=doesn't exist, no positive examples)
- **Characteristics**:
  - Continuous random timestamps
  - High stochasticity
  - Poor for learning patterns
- **Status**: **NOT RECOMMENDED**

#### **synthetic_sd_er** ❌ **HIGH RANDOMNESS**
- **Model**: Stochastic Diffusion on Erdős-Rényi network
- **Structure**: Similar to sd_ba
- **Randomness**: **HIGH** - Random graph + random diffusion
- **Status**: **NOT RECOMMENDED**

### 1.2 Triadic Closure Models
These datasets simulate social network formation based on triadic closure:

#### **triadic_perfect_* (various sizes)** ✅ **RECOMMENDED**
- **Model**: Perfect triadic closure with deterministic rules
- **Sizes Available**: small, medium, large, xl, dense, sparse
- **Randomness**: **VERY LOW** - Perfect deterministic patterns
- **Edge Labels**: Binary (1=exists, 0=doesn't exist)
- **Characteristics**:
  - Perfect triadic closure patterns
  - Highly predictable
  - Excellent for learning structural patterns
- **Status**: **READY FOR TESTING**

#### **triadic_fixed** ✅ **RECOMMENDED**
- **Model**: Fixed triadic closure patterns
- **Randomness**: **VERY LOW**
- **Status**: **READY FOR TESTING**

### 1.3 Generic Synthetic
#### **synthetic** (generic)
- **Model**: Generic synthetic graph generation
- **Randomness**: **UNKNOWN**
- **Status**: **NEEDS INVESTIGATION**

---

## 2. Randomness Assessment

### 2.1 Low Randomness (Recommended)
- **synthetic_ltm_ba**: Linear Threshold Model - deterministic activation
- **synthetic_icm_ba**: Independent Cascade Model - deterministic spreading
- **triadic_perfect_***: Perfect triadic closure - deterministic patterns

### 2.2 Medium Randomness (Moderate)
- **synthetic_cc_ws**: Community contagion - some randomness in community formation

### 2.3 High Randomness (Not Recommended)
- **synthetic_sd_ba/er**: Stochastic diffusion - continuous random processes

---

## 3. Testing Priority for GraphMamba

### 3.1 **HIGH PRIORITY** - Test Next
1. **synthetic_ltm_ba** - Low randomness, good structure, different from ICM
2. **triadic_perfect_medium** - Low randomness, different pattern type

### 3.2 **MEDIUM PRIORITY** - Test Later
3. **synthetic_cc_ws** - Moderate randomness, community-based patterns
4. **triadic_perfect_large** - Larger scale triadic patterns

### 3.3 **LOW PRIORITY** - Avoid for Now
5. **synthetic_sd_ba/er** - Too much randomness
6. **synthetic** - Unknown characteristics

---

## 4. Dataset Characteristics Summary

| Dataset | Type | Nodes | Edges | Timestamps | Randomness | Status |
|---------|------|-------|-------|------------|------------|---------|
| synthetic_icm_ba | Contagion | 80 | 79 | 29 | Low | ✅ Tested |
| synthetic_ltm_ba | Contagion | ~150 | 288 | Multiple | Low | 🔄 Ready |
| synthetic_cc_ws | Contagion | ~40 | 50 | 10 | Medium | 🔄 Ready |
| synthetic_sd_ba | Diffusion | ~30 | 23 | Continuous | High | ❌ Avoid |
| triadic_perfect_medium | Triadic | ~30 | 49 | 4 | Very Low | 🔄 Ready |
| triadic_perfect_large | Triadic | ~100 | ~200 | Multiple | Very Low | 🔄 Ready |

---

## 5. Recommendations

### 5.1 Immediate Testing
1. **synthetic_ltm_ba** - Different contagion model, low randomness
2. **triadic_perfect_medium** - Different pattern type, very low randomness

### 5.2 Why These Are Good Choices
- **Low randomness** = Better learning of patterns
- **Different models** = Test generalization across contagion types
- **Different sizes** = Test scalability
- **Clear patterns** = Easier to evaluate model performance

### 5.3 Expected Results
- **synthetic_ltm_ba**: Should achieve 70-85% AP (similar to ICM)
- **triadic_perfect_medium**: Should achieve 80-90% AP (very predictable patterns)

---

## 6. Next Steps

1. **Test synthetic_ltm_ba** with GraphMamba
2. **Test triadic_perfect_medium** with GraphMamba  
3. **Compare results** across different dataset types
4. **Update comprehensive report** with new findings

---

*Analysis completed: January 2025*  
*Based on examination of dataset files and explanations*

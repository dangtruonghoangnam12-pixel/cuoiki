import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import json
import logging
from typing import Dict, Any, List, Tuple
import os
import re
import queue
import threading
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle
import unicodedata
from difflib import SequenceMatcher
import unidecode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_search_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SearchException(Exception):
    """Custom exception for search system"""
    pass

class DataProcessingError(Exception):
    """Custom exception for data processing"""
    pass

class AdvancedTextProcessor:
    """Xử lý văn bản nâng cao với normalization, synonyms và fuzzy search"""
    
    def __init__(self):
        # Từ điển đồng nghĩa tiếng Việt cho đồ điện gia dụng
        self.synonyms = {
            # Loại thiết bị
            'máy lạnh': ['điều hòa', 'máy điều hòa', 'air conditioner', 'ac'],
            'tủ lạnh': ['tủ đông', 'refrigerator', 'fridge'],
            'nồi cơm điện': ['nồi cơm', 'rice cooker', 'nồi nấu cơm'],
            'máy giặt': ['washing machine', 'máy giặt quần áo'],
            'lò vi sóng': ['microwave', 'lò vi ba'],
            'quạt điện': ['quạt', 'fan', 'quạt gió'],
            'bếp điện': ['bếp từ', 'induction cooker', 'bếp điện từ'],
            'máy hút bụi': ['vacuum cleaner', 'máy hút'],
            'máy sấy tóc': ['hair dryer', 'máy sấy'],
            'ấm đun nước': ['bình đun nước', 'kettle', 'ấm siêu tốc'],
            'lò nướng': ['oven', 'lò nướng bánh'],
            'máy xay sinh tố': ['blender', 'máy xay', 'máy làm sinh tố'],
            'bàn ủi': ['iron', 'bàn là'],
            'máy rửa bát': ['dishwasher', 'máy rửa chén'],
            
            # Tính năng
            'tiết kiệm điện': ['eco', 'energy saving', 'inverter', 'tasking', 'tiết kiệm năng lượng'],
            'hẹn giờ': ['timer', 'đặt giờ', 'hẹn thời gian', 'chẹn giờ'],
            'điều khiển từ xa': ['remote', 'remote control', 'điều khiển xa'],
            'cảm ứng': ['touch', 'touchscreen', 'màn hình cảm ứng'],
            'chống dính': ['non-stick', 'không dính', 'chống bám'],
            'giữ ấm': ['keep warm', 'giữ nhiệt', 'duy trì nhiệt độ'],
            'tự động': ['auto', 'automatic', 'tự động hóa'],
            'khóa trẻ em': ['child lock', 'khóa an toàn', 'bảo vệ trẻ em'],
            'wifi': ['smart', 'thông minh', 'kết nối internet', 'iot'],
            'led': ['màn hình led', 'hiển thị led', 'đèn led'],
            'inox': ['thép không gỉ', 'stainless steel', 'thép ko gỉ'],
            
            # Thương hiệu (bao gồm lỗi chính tả thường gặp)
            'panasonic': ['panasonik', 'panasonic', 'pana'],
            'samsung': ['sam sung', 'samsung'],
            'lg': ['lg electronics', 'elgi'],
            'sharp': ['sarp', 'shapr'],
            'electrolux': ['electrolug', 'electrolux'],
            'philips': ['philip', 'phillps'],
            'toshiba': ['tosiba', 'toshiba'],
            'midea': ['midea', 'media'],
            'kangaroo': ['kangaruu', 'kăng ga ru'],
            'sunhouse': ['sun house', 'sunhose'],
            
            # Giá cả
            'rẻ': ['giá rẻ', 'tiết kiệm', 'bình dân', 'phải chăng', 'tốt'],
            'đắt': ['cao cấp', 'premium', 'sang trọng', 'chất lượng cao'],
            'triệu': ['tr', 'million', 'millions'],
            'nghìn': ['k', 'thousand', 'ngàn'],
            
            # Kích thước và công suất
            'nhỏ': ['compact', 'mini', 'nhỏ gọn', 'tiết kiệm không gian'],
            'lớn': ['big', 'to', 'khổ lớn', 'gia đình đông'],
            'công suất cao': ['mạnh', 'hiệu suất cao', 'nhanh'],
            'công suất thấp': ['yếu', 'tiết kiệm điện', 'ít điện'],
        }
        
        # Từ dừng tiếng Việt
        self.stop_words = {
            'và', 'hoặc', 'có', 'được', 'sẽ', 'đã', 'đang', 'là', 'của', 'cho', 'với', 
            'từ', 'đến', 'trong', 'ngoài', 'trên', 'dưới', 'về', 'theo', 'như', 'để',
            'này', 'đó', 'kia', 'nào', 'ai', 'gì', 'đâu', 'khi', 'nào', 'sao',
            'tôi', 'bạn', 'chúng', 'họ', 'nó', 'mình', 'ta', 'các', 'những', 'một',
            'cần', 'muốn', 'tìm', 'kiếm', 'mua', 'bán', 'xem', 'thích', 'ưa'
        }
    
    def normalize_text(self, text: str) -> str:
        """Bước 1: Chuẩn hóa văn bản"""
        if not isinstance(text, str):
            text = str(text)
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ dấu câu không cần thiết
        text = re.sub(r'[^\w\s.,()-]', ' ', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Chuẩn hóa Unicode
        text = unicodedata.normalize('NFC', text)
        
        return text
    
    def remove_accents(self, text: str) -> str:
        """Bỏ dấu tiếng Việt cho fuzzy search"""
        try:
            return unidecode.unidecode(text)
        except:
            # Fallback method if unidecode fails
            accented = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
            no_accent = "aaaaaaaaaaaaaaaaaeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
            translation = str.maketrans(accented, no_accent)
            return text.translate(translation)
    
    def expand_synonyms(self, text: str) -> str:
        """Bước 2: Mở rộng từ đồng nghĩa"""
        expanded_terms = []
        words = text.split()
        
        for word in words:
            if word not in self.stop_words:
                expanded_terms.append(word)
                
                # Tìm từ đồng nghĩa
                for key, synonyms in self.synonyms.items():
                    if word in synonyms or word == key:
                        expanded_terms.extend([key] + synonyms[:2])  # Thêm tối đa 2 từ đồng nghĩa
                        break
        
        return ' '.join(expanded_terms)
    
    def fuzzy_match(self, query: str, text: str, threshold: float = 0.6) -> float:
        """Bước 3: Fuzzy search với similarity score"""
        query_clean = self.remove_accents(self.normalize_text(query))
        text_clean = self.remove_accents(self.normalize_text(text))
        
        # Tính similarity score
        return SequenceMatcher(None, query_clean, text_clean).ratio()
    
    def process_search_query(self, query: str) -> Dict[str, Any]:
        """Xử lý query với 4 bước"""
        processed_query = {
            'original': query,
            'step1_normalized': '',
            'step2_with_synonyms': '',
            'step3_no_accents': '',
            'tokens': []
        }
        
        # Bước 1: Chuẩn hóa
        normalized = self.normalize_text(query)
        processed_query['step1_normalized'] = normalized
        
        # Bước 2: Mở rộng từ đồng nghĩa
        with_synonyms = self.expand_synonyms(normalized)
        processed_query['step2_with_synonyms'] = with_synonyms
        
        # Bước 3: Bỏ dấu cho fuzzy search
        no_accents = self.remove_accents(with_synonyms)
        processed_query['step3_no_accents'] = no_accents
        
        # Bước 4: Tách từ
        tokens = [token for token in with_synonyms.split() if token not in self.stop_words]
        processed_query['tokens'] = tokens
        
        return processed_query

class EnhancedPriceParser:
    """Parser giá cả được cải tiến"""
    
    def __init__(self):
        # Patterns cho việc parse giá cả
        self.price_patterns = [
            # "giá trên 2 triệu", "giá dưới 500k"
            r'giá\s*(trên|dưới|từ|đến|>=|<=|>|<)\s*(\d+(?:[.,]\d+)*)\s*(?:k|tr|triệu|nghìn|đồng|vnđ|đ)?',
            
            # "trên 2 triệu", "dưới 500 nghìn"
            r'(trên|dưới|từ|đến|>=|<=|>|<)\s*(\d+(?:[.,]\d+)*)\s*(?:k|tr|triệu|nghìn|đồng|vnđ|đ)',
            
            # "2 triệu trở lên", "500k trở xuống"
            r'(\d+(?:[.,]\d+)*)\s*(?:k|tr|triệu|nghìn|đồng|vnđ|đ)\s*(trở\s*lên|trở\s*xuống)',
            
            # Khoảng giá: "từ 1 đến 3 triệu"
            r'từ\s*(\d+(?:[.,]\d+)*)\s*(?:k|tr|triệu|nghìn)?\s*(?:đến|tới|-)\s*(\d+(?:[.,]\d+)*)\s*(?:k|tr|triệu|nghìn|đồng|vnđ|đ)',
            
            # Chỉ số: "2 triệu", "500k"
            r'(\d+(?:[.,]\d+)*)\s*(k|tr|triệu|nghìn|đồng|vnđ|đ)',
        ]
        
        # Mapping từ viết tắt
        self.unit_mapping = {
            'k': 1000,
            'tr': 1000000,
            'triệu': 1000000,
            'nghìn': 1000,
            'ngàn': 1000,
            'đồng': 1,
            'vnđ': 1,
            'đ': 1
        }
        
        # Mapping toán tử
        self.operator_mapping = {
            'trên': '>',
            'dưới': '<',
            'từ': '>=',
            'đến': '<=',
            'trở lên': '>=',
            'trở xuống': '<=',
            '>=': '>=',
            '<=': '<=',
            '>': '>',
            '<': '<'
        }
    
    def parse_price_conditions(self, query: str) -> Dict[str, Any]:
        """Parse điều kiện giá từ query"""
        conditions = {
            'price_min': None,
            'price_max': None,
            'price_exact': None,
            'parsed_info': []
        }
        
        query_lower = query.lower()
        
        for pattern in self.price_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            
            for match in matches:
                groups = match.groups()
                info = {'pattern': pattern, 'match': match.group(0), 'groups': groups}
                
                try:
                    if len(groups) == 2:  # operator + price
                        operator = groups[0].strip()
                        price_str = groups[1].strip()
                        unit = self._extract_unit_from_match(match.group(0))
                        
                        price_value = self._parse_price_value(price_str, unit)
                        
                        if price_value > 0:
                            if operator in ['trên', '>', '>=', 'từ', 'trở lên']:
                                conditions['price_min'] = price_value
                            elif operator in ['dưới', '<', '<=', 'đến', 'trở xuống']:
                                conditions['price_max'] = price_value
                            
                            info['parsed_price'] = price_value
                            info['operator'] = operator
                            
                    elif len(groups) == 3:  # price + unit + modifier
                        price_str = groups[0].strip()
                        unit = groups[1].strip() if groups[1] else None
                        modifier = groups[2].strip() if groups[2] else None
                        
                        price_value = self._parse_price_value(price_str, unit)
                        
                        if price_value > 0:
                            if modifier and 'lên' in modifier:
                                conditions['price_min'] = price_value
                            elif modifier and 'xuống' in modifier:
                                conditions['price_max'] = price_value
                            else:
                                conditions['price_exact'] = price_value
                            
                            info['parsed_price'] = price_value
                            info['modifier'] = modifier
                    
                    elif len(groups) == 4:  # từ X đến Y
                        price1_str = groups[0].strip()
                        unit1 = groups[1].strip() if groups[1] else 'đồng'
                        price2_str = groups[2].strip()
                        unit2 = groups[3].strip() if groups[3] else unit1
                        
                        price1 = self._parse_price_value(price1_str, unit1)
                        price2 = self._parse_price_value(price2_str, unit2)
                        
                        if price1 > 0 and price2 > 0:
                            conditions['price_min'] = min(price1, price2)
                            conditions['price_max'] = max(price1, price2)
                            
                            info['parsed_price_min'] = price1
                            info['parsed_price_max'] = price2
                    
                    conditions['parsed_info'].append(info)
                    
                except Exception as e:
                    logger.warning(f"Error parsing price pattern: {e}")
                    continue
        
        return conditions
    
    def _extract_unit_from_match(self, match_text: str) -> str:
        """Trích xuất đơn vị từ text match"""
        for unit in self.unit_mapping.keys():
            if unit in match_text.lower():
                return unit
        return 'đồng'
    
    def _parse_price_value(self, price_str: str, unit: str = None) -> int:
        """Chuyển đổi string giá thành số"""
        try:
            # Làm sạch string
            clean_price = re.sub(r'[^\d.,]', '', price_str)
            
            # Xử lý dấu phẩy và chấm
            if ',' in clean_price and '.' in clean_price:
                # Có cả phẩy và chấm -> phẩy là thousand separator
                clean_price = clean_price.replace(',', '')
            elif ',' in clean_price:
                # Chỉ có phẩy -> kiểm tra vị trí
                parts = clean_price.split(',')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    # Phẩy là decimal separator
                    clean_price = clean_price.replace(',', '.')
                else:
                    # Phẩy là thousand separator
                    clean_price = clean_price.replace(',', '')
            
            # Chuyển thành float
            price_value = float(clean_price)
            
            # Áp dụng đơn vị
            if unit and unit in self.unit_mapping:
                price_value *= self.unit_mapping[unit]
            
            return int(price_value)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot parse price '{price_str}' with unit '{unit}': {e}")
            return 0
    
    def check_price_match(self, product_price_str: str, conditions: Dict[str, Any]) -> bool:
        """Kiểm tra giá sản phẩm có phù hợp với điều kiện không"""
        try:
            # Parse giá sản phẩm
            product_price = self._parse_product_price(product_price_str)
            
            if product_price == 0:
                return True  # Không loại bỏ nếu không parse được giá
            
            # Kiểm tra điều kiện
            if conditions.get('price_min') is not None:
                if product_price < conditions['price_min']:
                    return False
            
            if conditions.get('price_max') is not None:
                if product_price > conditions['price_max']:
                    return False
            
            if conditions.get('price_exact') is not None:
                # Cho phép sai lệch 10%
                exact_price = conditions['price_exact']
                if abs(product_price - exact_price) / exact_price > 0.1:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking price match: {e}")
            return True
    
    def _parse_product_price(self, price_str: str) -> int:
        """Parse giá từ dữ liệu sản phẩm"""
        try:
            if not price_str or price_str in ['', 'Không có thông tin', 'N/A']:
                return 0
            
            # Tìm số đầu tiên trong string
            price_match = re.search(r'(\d+(?:[.,]\d+)*)', str(price_str))
            if not price_match:
                return 0
            
            price_num = price_match.group(1)
            
            # Xác định đơn vị dựa trên context
            price_str_lower = str(price_str).lower()
            if 'triệu' in price_str_lower or 'tr' in price_str_lower:
                unit = 'triệu'
            elif 'nghìn' in price_str_lower or 'k' in price_str_lower:
                unit = 'nghìn'
            else:
                # Heuristic: nếu số lớn hơn 1000 và không có đơn vị -> có thể là VNĐ
                num_value = float(price_num.replace(',', '.'))
                if num_value > 1000:
                    unit = 'đồng'
                else:
                    unit = 'triệu'  # Giả định mặc định cho đồ gia dụng
            
            return self._parse_price_value(price_num, unit)
            
        except Exception as e:
            logger.warning(f"Error parsing product price '{price_str}': {e}")
            return 0

class VectorDatabase:
    """Vector database for semantic search - updated"""
    
    def __init__(self, model_path: str = "search_model.pkl"):
        self.vectorizer = None
        self.product_vectors = None
        self.products = []
        self.model_path = model_path
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.is_trained = False
        self.text_processor = AdvancedTextProcessor()
        
    def normalize_vietnamese_text(self, text: str) -> str:
        """Chuẩn hóa văn bản tiếng Việt"""
        return self.text_processor.normalize_text(text)
    
    def create_product_text(self, product: Dict[str, Any]) -> str:
        """Tạo văn bản tìm kiếm từ thông tin sản phẩm"""
        try:
            text_parts = []
            
            # Tên sản phẩm (trọng số cao)
            name = self.normalize_vietnamese_text(product.get('ten_san_pham', ''))
            processed_name = self.text_processor.expand_synonyms(name)
            text_parts.extend([processed_name] * 3)
            
            # Thương hiệu (trọng số cao)
            brand = self.normalize_vietnamese_text(product.get('thuong_hieu', ''))
            processed_brand = self.text_processor.expand_synonyms(brand)
            text_parts.extend([processed_brand] * 2)
            
            # Tính năng và tiện ích
            features = product.get('tien_ich', [])
            if isinstance(features, list):
                for feature in features:
                    processed_feature = self.text_processor.expand_synonyms(
                        self.normalize_vietnamese_text(feature)
                    )
                    text_parts.append(processed_feature)
            
            # Đặc tính bổ sung
            dac_tinh = self.normalize_vietnamese_text(product.get('dac_tinh', ''))
            if dac_tinh and dac_tinh != 'không có thông tin':
                processed_dac_tinh = self.text_processor.expand_synonyms(dac_tinh)
                text_parts.append(processed_dac_tinh)
            
            # Loại sản phẩm
            type_info = self.normalize_vietnamese_text(product.get('loai_noi', ''))
            if type_info and type_info != 'không có thông tin':
                processed_type = self.text_processor.expand_synonyms(type_info)
                text_parts.append(processed_type)
            
            # Công suất
            power = self.normalize_vietnamese_text(product.get('cong_suat', ''))
            if power and power != 'không có thông tin':
                text_parts.append(power)
            
            return ' '.join([part for part in text_parts if part.strip()])
            
        except Exception as e:
            logger.warning(f"Error creating product text: {str(e)}")
            return self.normalize_vietnamese_text(product.get('ten_san_pham', ''))
    
    def train(self, products: List[Dict[str, Any]]):
        """Huấn luyện vector database"""
        try:
            if not products:
                raise ValueError("No products provided for training")
            
            self.products = products
            logger.info(f"Training vector database with {len(products)} products")
            
            # Tạo văn bản đại diện cho sản phẩm
            product_texts = []
            for product in products:
                text = self.create_product_text(product)
                product_texts.append(text)
            
            # Khởi tạo TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=None,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.95,
                lowercase=True,
                token_pattern=r'(?u)\b\w+\b'
            )
            
            # Fit và transform
            tfidf_matrix = self.vectorizer.fit_transform(product_texts)
            
            # Giảm chiều để tăng hiệu suất
            self.product_vectors = self.svd.fit_transform(tfidf_matrix)
            
            self.is_trained = True
            logger.info(f"✅ Vector database trained with {self.product_vectors.shape[1]} dimensions")
            
            self.save_model()
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise SearchException(f"Failed to train vector database: {str(e)}")
    
    def save_model(self):
        """Lưu model đã huấn luyện"""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'svd': self.svd,
                'product_vectors': self.product_vectors,
                'products': self.products,
                'text_processor': self.text_processor
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Save model error: {str(e)}")
    
    def load_model(self):
        """Load model từ disk"""
        try:
            if not os.path.exists(self.model_path):
                return False
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.svd = model_data['svd']
            self.product_vectors = model_data['product_vectors']
            self.products = model_data['products']
            if 'text_processor' in model_data:
                self.text_processor = model_data['text_processor']
            
            self.is_trained = True
            logger.info("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Load model error: {str(e)}")
            return False
    
    def semantic_search(self, query: str, top_k: int = 50, threshold: float = 0.1) -> List[Tuple[Dict[str, Any], float]]:
        """Tìm kiếm ngữ nghĩa sử dụng vector similarity"""
        try:
            if not self.is_trained:
                raise SearchException("Vector database not trained")
            
            # Xử lý query qua 4 bước
            processed_query = self.text_processor.process_search_query(query)
            search_text = processed_query['step2_with_synonyms']
            
            logger.info(f"Search steps: {processed_query}")
            
            # Transform query thành vector
            query_tfidf = self.vectorizer.transform([search_text])
            query_vector = self.svd.transform(query_tfidf)
            
            # Tính similarity
            similarities = cosine_similarity(query_vector, self.product_vectors)[0]
            
            # Kết hợp với fuzzy search
            results = []
            for i, similarity in enumerate(similarities):
                product = self.products[i]
                
                # Vector similarity
                if similarity >= threshold:
                    final_score = similarity
                else:
                    # Fallback: fuzzy search
                    product_text = self.create_product_text(product)
                    fuzzy_score = self.text_processor.fuzzy_match(query, product_text)
                    
                    # Chỉ chấp nhận nếu fuzzy score đủ cao
                    if fuzzy_score >= 0.3:
                        final_score = fuzzy_score * 0.7  # Giảm trọng số fuzzy
                    else:
                        continue
                
                results.append((product, final_score))
            
            # Sắp xếp theo score
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            raise SearchException(f"Semantic search failed: {str(e)}")

class SmartQueryParser:
    """Parser query thông minh với price parser cải tiến"""
    
    def __init__(self):
        self.price_parser = EnhancedPriceParser()
        self.text_processor = AdvancedTextProcessor()
        
        # Enhanced feature keywords
        self.feature_keywords = {
            'hẹn giờ': ['hẹn giờ', 'timer', 'hẹn thời gian', 'đặt giờ'],
            'giữ ấm': ['giữ ấm', 'keep warm', 'giữ nhiệt'],
            'chống dính': ['chống dính', 'non-stick', 'không dính'],
            'khóa trẻ em': ['khóa trẻ em', 'child lock', 'an toàn trẻ em'],
            'cảm ứng': ['cảm ứng', 'touch', 'màn hình cảm ứng'],
            'led': ['led', 'màn hình led', 'hiển thị led'],
            'inox': ['inox', 'thép không gỉ', 'stainless steel'],
            'remote': ['remote', 'điều khiển từ xa'],
            'wifi': ['wifi', 'smart', 'thông minh', 'kết nối'],
            'tiết kiệm điện': ['tiết kiệm điện', 'inverter', 'eco'],
            'tự động': ['tự động', 'automatic', 'auto'],
        }
        
        self.brand_keywords = [
            'panasonic', 'sharp', 'toshiba', 'tiger', 'zojirushi', 'cuckoo', 
            'sunhouse', 'electrolux', 'philips', 'tefal', 'midea', 'supor',
            'samsung', 'lg', 'daikin', 'mitsubishi', 'hitachi', 'fujitsu',
            'bosch', 'siemens', 'whirlpool', 'haier', 'aqua', 'sanyo',
            'kangaroo', 'bluestone', 'elmich', 'happy cook'
        ]
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query thành structured filters"""
        filters = {'keyword': '', 'extracted_conditions': [], 'search_strategy': []}
        
        # Xử lý query qua text processor
        processed = self.text_processor.process_search_query(query)
        filters['processed_query'] = processed
        filters['search_strategy'].append('normalized_and_synonyms')
        
        # Extract price conditions với parser mới
        price_conditions = self.price_parser.parse_price_conditions(query)
        if price_conditions.get('price_min') or price_conditions.get('price_max'):
            filters.update(price_conditions)
            filters['search_strategy'].append('price_filtering')
        
        # Extract brand
        brand = self._extract_brand(query)
        if brand:
            filters['brand'] = brand
            filters['search_strategy'].append('brand_filtering')
        
        # Extract features
        features, remaining_query = self._extract_features(processed['step2_with_synonyms'])
        if features:
            filters['required_features'] = features
            filters['search_strategy'].append('feature_matching')
        
        # Clean remaining query
        remaining_query = self._clean_query(remaining_query or processed['step2_with_synonyms'])
        if remaining_query.strip():
            filters['keyword'] = remaining_query
            
            # Determine search strategy based on query length
            if len(remaining_query.split()) >= 3:
                filters['search_strategy'].append('vector_search')
            else:
                filters['search_strategy'].append('fuzzy_search')
        
        return filters
    
    def _extract_brand(self, query: str) -> str:
        """Extract brand từ query"""
        query_lower = query.lower()
        for brand in self.brand_keywords:
            if brand in query_lower:
                return brand.title()
        return None
    
    def _extract_features(self, query: str) -> tuple:
        """Extract required features và return remaining query"""
        required_features = []
        remaining_query = query
        
        for feature, keywords in self.feature_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    required_features.append(feature)
                    remaining_query = remaining_query.replace(keyword, ' ')
                    break
        
        return required_features, remaining_query
    
    def _clean_query(self, query: str) -> str:
        """Clean query by removing extracted conditions"""
        # Remove price expressions
        for pattern in self.price_parser.price_patterns:
            query = re.sub(pattern, ' ', query, flags=re.IGNORECASE)
        
        # Remove brands
        for brand in self.brand_keywords:
            query = query.replace(brand, ' ')
        
        # Remove common words
        words = query.split()
        words = [word for word in words if word not in self.text_processor.stop_words and len(word.strip()) > 1]
        
        return ' '.join(words).strip()

class ApplianceDatabase:
    """Enhanced database với 4-step search system"""
    
    def __init__(self, json_file: str = None):
        self.products = []
        self.metadata = {}
        self.is_loaded = False
        self.json_file = json_file
        self.query_parser = SmartQueryParser()
        self.vector_db = VectorDatabase()
        self.text_processor = AdvancedTextProcessor()
        self.price_parser = EnhancedPriceParser()
        
        if json_file:
            self._load_data()
    
    def load_file(self, json_file: str):
        """Load data from JSON file"""
        self.json_file = json_file
        self._load_data()
    
    def _load_data(self):
        """Load và validate data từ JSON file"""
        if not self.json_file:
            logger.warning("No JSON file specified")
            return
            
        try:
            if not os.path.exists(self.json_file):
                raise DataProcessingError(f"File not found: {self.json_file}")
            
            if os.path.getsize(self.json_file) == 0:
                raise DataProcessingError(f"File is empty: {self.json_file}")
            
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if 'products' in data:
                self.products = data['products']
                self.metadata = data.get('metadata', {})
            elif isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
                self.products = list(data.values())
                self.metadata = {"total_products": len(self.products)}
            elif isinstance(data, list):
                self.products = data
                self.metadata = {"total_products": len(data)}
            else:
                raise DataProcessingError("Invalid JSON structure")
            
            if not self.products:
                logger.warning("No products found in JSON file")
                return
            
            # Train vector database
            logger.info("Training vector database...")
            self.vector_db.train(self.products)
            
            self.is_loaded = True
            logger.info(f"✅ Successfully loaded {len(self.products)} products")
            
        except json.JSONDecodeError as e:
            raise DataProcessingError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise DataProcessingError(f"Failed to load data: {str(e)}")
    
    def smart_search(self, query: str) -> List[Dict[str, Any]]:
        """
        4-step intelligent search system:
        1. Normalize text + synonyms
        2. Check exact matches
        3. Fuzzy search if needed
        4. Vector search for complex queries
        """
        try:
            if not self.is_loaded:
                raise SearchException("Data not loaded")
            
            # Parse query
            filters = self.query_parser.parse_query(query)
            logger.info(f"Search strategy: {filters.get('search_strategy', [])}")
            
            results = []
            search_strategies = filters.get('search_strategy', [])
            
            # Bước 1-2: Exact matching với synonyms và normalization
            if 'normalized_and_synonyms' in search_strategies:
                exact_results = self._exact_search(filters)
                results.extend(exact_results)
                logger.info(f"Exact search found {len(exact_results)} results")
            
            # Bước 3: Fuzzy search nếu cần
            if len(results) < 10 and 'fuzzy_search' in search_strategies:
                fuzzy_results = self._fuzzy_search(filters)
                # Merge và loại bỏ duplicate
                results = self._merge_results(results, fuzzy_results)
                logger.info(f"After fuzzy search: {len(results)} results")
            
            # Bước 4: Vector search cho query phức tạp
            if 'vector_search' in search_strategies:
                vector_results = self.vector_db.semantic_search(query)
                vector_products = [result[0] for result in vector_results]
                results = self._merge_results(results, vector_products)
                logger.info(f"After vector search: {len(results)} results")
            
            # Apply additional filters
            filtered_results = self._apply_filters(results, filters)
            logger.info(f"Final results after filtering: {len(filtered_results)} products")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Smart search error: {str(e)}")
            raise SearchException(f"Search failed: {str(e)}")
    
    def _exact_search(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Exact matching với normalized text và synonyms"""
        results = []
        keyword = filters.get('keyword', '').lower()
        
        if not keyword:
            return results
        
        # Expand keyword với synonyms
        expanded_keyword = self.text_processor.expand_synonyms(keyword)
        search_terms = set(expanded_keyword.split())
        
        for product in self.products:
            # Tạo searchable text
            searchable_text = self._create_searchable_text(product).lower()
            
            # Check exact matches
            matches = 0
            for term in search_terms:
                if term in searchable_text:
                    matches += 1
            
            # Require at least 50% terms to match
            if matches >= len(search_terms) * 0.5:
                results.append(product)
        
        return results
    
    def _fuzzy_search(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fuzzy search với similarity threshold"""
        results = []
        keyword = filters.get('keyword', '')
        
        if not keyword:
            return results
        
        for product in self.products:
            searchable_text = self._create_searchable_text(product)
            
            # Calculate fuzzy similarity
            similarity = self.text_processor.fuzzy_match(keyword, searchable_text)
            
            if similarity >= 0.4:  # 40% similarity threshold
                product_copy = product.copy()
                product_copy['_fuzzy_score'] = similarity
                results.append(product_copy)
        
        # Sort by similarity
        results.sort(key=lambda x: x.get('_fuzzy_score', 0), reverse=True)
        return results
    
    def _create_searchable_text(self, product: Dict[str, Any]) -> str:
        """Tạo text có thể search được từ product"""
        text_parts = []
        
        # Key fields
        text_parts.append(str(product.get('ten_san_pham', '')))
        text_parts.append(str(product.get('thuong_hieu', '')))
        text_parts.append(str(product.get('dac_tinh', '')))
        text_parts.append(str(product.get('loai_noi', '')))
        
        # Features
        features = product.get('tien_ich', [])
        if isinstance(features, list):
            text_parts.extend(features)
        
        return ' '.join(text_parts)
    
    def _merge_results(self, existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """Merge results và loại bỏ duplicates"""
        # Simple deduplication based on product name
        seen_names = set()
        merged = []
        
        for product in existing + new:
            name = product.get('ten_san_pham', '')
            if name not in seen_names:
                seen_names.add(name)
                merged.append(product)
        
        return merged
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply price, brand và feature filters"""
        filtered = []
        
        for product in results:
            # Price filter
            if not self._check_price_filter(product, filters):
                continue
            
            # Brand filter
            if not self._check_brand_filter(product, filters):
                continue
            
            # Feature filter
            if not self._check_feature_filter(product, filters):
                continue
            
            filtered.append(product)
        
        return filtered
    
    def _check_price_filter(self, product: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check price conditions với enhanced parser"""
        price_conditions = {
            'price_min': filters.get('price_min'),
            'price_max': filters.get('price_max'),
            'price_exact': filters.get('price_exact')
        }
        
        # Skip if no price conditions
        if not any(price_conditions.values()):
            return True
        
        return self.price_parser.check_price_match(
            product.get('gia_tien', ''), 
            price_conditions
        )
    
    def _check_brand_filter(self, product: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check brand filter"""
        filter_brand = filters.get('brand')
        if not filter_brand or filter_brand == 'Tất cả':
            return True
        
        product_brand = str(product.get('thuong_hieu', '')).lower()
        return filter_brand.lower() in product_brand
    
    def _check_feature_filter(self, product: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check required features"""
        required_features = filters.get('required_features', [])
        if not required_features:
            return True
        
        # Combine all product text for feature checking
        product_text = self._create_searchable_text(product).lower()
        
        for required_feature in required_features:
            feature_found = False
            keywords = self.query_parser.feature_keywords.get(required_feature, [required_feature])
            
            for keyword in keywords:
                if keyword.lower() in product_text:
                    feature_found = True
                    break
            
            if not feature_found:
                return False
        
        return True

class ApplianceSearchGUI:
    """Enhanced GUI với 4-step search system"""
    
    def __init__(self, json_file=None):
        self.root = tk.Tk()
        self.db = ApplianceDatabase()
        self.search_queue = queue.Queue()
        self.json_file = json_file
        self._setup_window()
        self._create_widgets()
        if json_file:
            self._initialize_database()
    
    def _setup_window(self):
        """Configure main window"""
        try:
            self.root.title("🔍 Hệ thống tìm kiếm thông minh 4 bước - Đồ điện gia dụng")
            self.root.geometry("1400x900")
            self.root.configure(bg='#f0f0f0')
            
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
            
        except Exception as e:
            logger.error(f"Window setup error: {str(e)}")
            messagebox.showerror("Error", f"Failed to setup GUI: {str(e)}")
    
    def _create_widgets(self):
        """Create GUI widgets"""
        try:
            # Header
            header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
            header_frame.pack(fill='x', padx=10, pady=(10, 0))
            header_frame.pack_propagate(False)
            
            title_label = tk.Label(header_frame, 
                                 text="🧠 TÌM KIẾM THÔNG MINH 4 BƯỚC", 
                                 font=('Arial', 18, 'bold'), 
                                 bg='#2c3e50', 
                                 fg='white')
            title_label.pack(expand=True)
            
            # Main container
            main_frame = tk.Frame(self.root, bg='#f0f0f0')
            main_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Left panel
            self._create_search_panel(main_frame)
            
            # Right panel
            self._create_results_panel(main_frame)
            
            # Bottom panel
            self._create_status_panel()
            
        except Exception as e:
            logger.error(f"Widget creation error: {str(e)}")
            messagebox.showerror("Error", f"Failed to create widgets: {str(e)}")
    
    def _create_search_panel(self, parent):
        """Create enhanced search panel"""
        try:
            search_frame = tk.LabelFrame(parent, 
                                       text="🔍 Tìm kiếm thông minh", 
                                       font=('Arial', 12, 'bold'), 
                                       bg='#f0f0f0')
            search_frame.pack(side='left', fill='y', padx=(0, 10))
            
            # File selection
            file_frame = tk.Frame(search_frame, bg='#f0f0f0')
            file_frame.pack(padx=10, pady=(10, 15))
            
            tk.Button(file_frame, 
                     text="📂 Chọn file JSON", 
                     command=self._select_json_file,
                     bg='#3498db', 
                     fg='white', 
                     font=('Arial', 10),
                     width=20).pack()
            
            self.current_file_var = tk.StringVar(value="Chưa chọn file")
            file_label = tk.Label(search_frame, 
                                textvariable=self.current_file_var,
                                font=('Arial', 8), 
                                bg='#f0f0f0',
                                fg='#666666',
                                wraplength=200)
            file_label.pack(padx=10, pady=(0, 10))
            
            # Smart search section
            smart_frame = tk.LabelFrame(search_frame, 
                                      text="🧠 Tìm kiếm thông minh (4 bước)", 
                                      font=('Arial', 11, 'bold'), 
                                      bg='#f0f0f0',
                                      fg='#2c3e50')
            smart_frame.pack(fill='x', padx=10, pady=(0, 15))
            
            # Search input
            tk.Label(smart_frame, 
                   text="Nhập yêu cầu tìm kiếm:", 
                   font=('Arial', 9, 'bold'), 
                   bg='#f0f0f0').pack(anchor='w', padx=10, pady=(10, 5))
            
            self.search_entry = tk.Text(smart_frame, height=4, width=25, wrap=tk.WORD, font=('Arial', 9))
            self.search_entry.pack(padx=10, pady=(0, 5))
            
            # Example text
            example_text = """Ví dụ về tìm kiếm thông minh:
• 'máy lạnh Samsung trên 5 triệu có wifi'
• 'nồi cơm điện dưới 2 triệu giữ ấm'  
• 'tủ lạnh Panasonic tiết kiệm điện'
• 'máy giặt inverter có hẹn giờ'"""
            
            example_label = tk.Label(smart_frame, 
                                   text=example_text, 
                                   font=('Arial', 7, 'italic'), 
                                   bg='#f0f0f0',
                                   fg='#7f8c8d',
                                   justify='left')
            example_label.pack(anchor='w', padx=10, pady=(0, 10))
            
            # Search button
            search_btn = tk.Button(smart_frame, 
                                 text="🔍 Tìm kiếm thông minh", 
                                 command=self._smart_search,
                                 bg='#27ae60', 
                                 fg='white', 
                                 font=('Arial', 10, 'bold'),
                                 width=20)
            search_btn.pack(padx=10, pady=(0, 10))
            
            # Advanced filters
            self._create_advanced_filters(search_frame)
            
        except Exception as e:
            logger.error(f"Search panel error: {str(e)}")
            raise
    
    def _create_advanced_filters(self, parent):
        """Create advanced filter section"""
        try:
            filter_frame = tk.LabelFrame(parent, 
                                       text="⚙️ Bộ lọc nâng cao", 
                                       font=('Arial', 11, 'bold'), 
                                       bg='#f0f0f0',
                                       fg='#2c3e50')
            filter_frame.pack(fill='x', padx=10, pady=(0, 10))
            
            # Price range with examples
            price_label = tk.Label(filter_frame, 
                                 text="Khoảng giá (VD: 'trên 2 triệu', 'dưới 500k'):", 
                                 font=('Arial', 9, 'bold'), 
                                 bg='#f0f0f0')
            price_label.pack(anchor='w', padx=10, pady=(10, 5))
            
            self.price_example_var = tk.StringVar()
            price_example_entry = tk.Entry(filter_frame, 
                                         textvariable=self.price_example_var,
                                         width=25,
                                         font=('Arial', 9))
            price_example_entry.pack(padx=10, pady=(0, 5))
            
            # Brand filter
            tk.Label(filter_frame, 
                   text="Thương hiệu:", 
                   font=('Arial', 10, 'bold'), 
                   bg='#f0f0f0').pack(anchor='w', padx=10, pady=(10, 5))
            
            self.brand_var = tk.StringVar(value="Tất cả")
            self.brand_combo = ttk.Combobox(filter_frame, 
                                          textvariable=self.brand_var, 
                                          width=22,
                                          values=["Tất cả"])
            self.brand_combo.pack(padx=10, pady=(0, 10))
            
            # Action buttons
            button_frame = tk.Frame(filter_frame, bg='#f0f0f0')
            button_frame.pack(padx=10, pady=10)
            
            clear_btn = tk.Button(button_frame, 
                                text="🗑️ Xóa bộ lọc", 
                                command=self._clear_filters,
                                bg='#95a5a6', 
                                fg='white', 
                                font=('Arial', 10),
                                width=20)
            clear_btn.pack(pady=(0, 5))
            
            retrain_btn = tk.Button(button_frame, 
                                  text="🔄 Huấn luyện lại", 
                                  command=self._retrain_model,
                                  bg='#e67e22', 
                                  fg='white', 
                                  font=('Arial', 9),
                                  width=20)
            retrain_btn.pack()
            
        except Exception as e:
            logger.error(f"Advanced filters error: {str(e)}")
    
    def _smart_search(self):
        """Perform 4-step smart search"""
        try:
            if not self.db or not self.db.is_loaded:
                messagebox.showwarning("Warning", "Vui lòng chọn và tải file JSON trước")
                return
            
            # Get search query
            search_query = self.search_entry.get("1.0", tk.END).strip()
            price_condition = self.price_example_var.get().strip()
            
            # Combine query with price condition
            full_query = f"{search_query} {price_condition}".strip()
            
            if not full_query:
                messagebox.showwarning("Warning", "Vui lòng nhập từ khóa tìm kiếm")
                return
            
            self._update_status(f"Đang tìm kiếm thông minh: '{full_query}'...")
            self.progress.start()
            
            def smart_search():
                try:
                    results = self.db.smart_search(full_query)
                    self.root.after(0, lambda: self._on_smart_search_completed(results, full_query))
                except Exception as e:
                    self.root.after(0, lambda: self._on_search_error(str(e)))
            
            threading.Thread(target=smart_search, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Smart search error: {str(e)}")
            messagebox.showerror("Error", f"Smart search failed: {str(e)}")
    
    def _on_smart_search_completed(self, results, original_query):
        """Smart search completed callback"""
        try:
            self.progress.stop()
            self._update_status(f"Tìm thấy {len(results)} sản phẩm cho: '{original_query}'")
            self._display_smart_results(results, original_query)
        except Exception as e:
            logger.error(f"Smart search completion error: {str(e)}")
            messagebox.showerror("Error", f"Failed to display results: {str(e)}")
    
    def _create_results_panel(self, parent):
        """Create results display panel"""
        try:
            results_frame = tk.LabelFrame(parent, 
                                        text="📋 Kết quả tìm kiếm", 
                                        font=('Arial', 12, 'bold'), 
                                        bg='#f0f0f0')
            results_frame.pack(side='right', fill='both', expand=True)
            
            self.results_text = scrolledtext.ScrolledText(results_frame, 
                                                        wrap=tk.WORD, 
                                                        height=40, 
                                                        font=('Courier', 10))
            self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Text tags
            self.results_text.tag_configure('title', font=('Arial', 12, 'bold'), foreground='#e74c3c')
            self.results_text.tag_configure('brand', font=('Arial', 10, 'bold'), foreground='#3498db')
            self.results_text.tag_configure('price', font=('Arial', 11, 'bold'), foreground='#27ae60')
            self.results_text.tag_configure('spec', font=('Arial', 9), foreground='#555555')
            self.results_text.tag_configure('highlight', font=('Arial', 9, 'bold'), foreground='#e74c3c', background='#fff3cd')
            self.results_text.tag_configure('query_info', font=('Arial', 10, 'bold'), foreground='#2c3e50', background='#ecf0f1')
            self.results_text.tag_configure('strategy', font=('Arial', 9, 'italic'), foreground='#8e44ad')
            
        except Exception as e:
            logger.error(f"Results panel error: {str(e)}")
            raise
    
    def _display_smart_results(self, results, search_query):
        """Display smart search results với search strategy info"""
        try:
            self.results_text.delete(1.0, tk.END)
            
            if not results:
                self.results_text.insert(tk.END, f"Tìm thấy {len(results)} sản phẩm\n\n", 'query_info')
            self.results_text.insert(tk.END, f"{'='*100}\n\n", 'spec')
            
            for i, product in enumerate(results, 1):
                # Product header
                self.results_text.insert(tk.END, f"{'='*80}\n", 'spec')
                self.results_text.insert(tk.END, f"🏠 SẢN PHẨM {i}", 'title')
                
                # Show search scores if available
                if '_fuzzy_score' in product:
                    score = product['_fuzzy_score']
                    self.results_text.insert(tk.END, f" - Điểm fuzzy: {score:.3f}\n", 'strategy')
                elif '_similarity_score' in product:
                    score = product['_similarity_score']
                    self.results_text.insert(tk.END, f" - Điểm vector: {score:.3f}\n", 'strategy')
                else:
                    self.results_text.insert(tk.END, "\n", 'title')
                
                self.results_text.insert(tk.END, f"{'='*80}\n", 'spec')
                
                # Product details
                self.results_text.insert(tk.END, f"📝 Tên: ", 'spec')
                self.results_text.insert(tk.END, f"{product.get('ten_san_pham', 'Không có thông tin')}\n\n", 'title')
                
                self.results_text.insert(tk.END, f"🏷️  Thương hiệu: ", 'spec')
                self.results_text.insert(tk.END, f"{product.get('thuong_hieu', 'Không có thông tin')}\n", 'brand')
                
                # Price with highlighting if matched price condition
                self.results_text.insert(tk.END, f"💰 Giá: ", 'spec')
                price_display = product.get('gia_tien', 'Không có thông tin')
                if filters.get('price_min') or filters.get('price_max'):
                    self.results_text.insert(tk.END, f"{price_display}\n", 'highlight')
                else:
                    self.results_text.insert(tk.END, f"{price_display}\n", 'price')
                
                # Specifications
                self.results_text.insert(tk.END, f"📊 Thông số:\n", 'spec')
                self.results_text.insert(tk.END, f"   • Công suất: {product.get('cong_suat', 'Không có thông tin')}\n", 'spec')
                self.results_text.insert(tk.END, f"   • Bảo hành: {product.get('bao_hanh', 'Không có thông tin')}\n", 'spec')
                self.results_text.insert(tk.END, f"   • Loại: {product.get('loai_noi', 'Không có thông tin')}\n", 'spec')
                
                # Features with highlighting
                features = product.get('tien_ich', [])
                if features and isinstance(features, list):
                    self.results_text.insert(tk.END, f"⚡ Tiện ích: ", 'spec')
                    features_text = ', '.join(features)
                    self.results_text.insert(tk.END, f"{features_text}\n", 'highlight')
                
                # Additional characteristics
                dac_tinh = product.get('dac_tinh', '')
                if dac_tinh and dac_tinh != 'Không có thông tin':
                    self.results_text.insert(tk.END, f"🔧 Đặc tính: ", 'spec')
                    self.results_text.insert(tk.END, f"{dac_tinh}\n", 'highlight')
                
                self.results_text.insert(tk.END, "\n")
            
            # Scroll to top
            self.results_text.see(1.0)
            
        except Exception as e:
            logger.error(f"Display smart results error: {str(e)}")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Lỗi hiển thị kết quả: {str(e)}")
    
    def _select_json_file(self):
        """Select JSON file dialog"""
        try:
            file_path = filedialog.askopenfilename(
                title="Chọn file JSON",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                self.json_file = file_path
                filename = os.path.basename(file_path)
                self.current_file_var.set(f"File: {filename}")
                self._initialize_database()
                
        except Exception as e:
            logger.error(f"File selection error: {str(e)}")
            messagebox.showerror("Error", f"Failed to select file: {str(e)}")
    
    def _update_filter_options(self):
        """Update filter dropdown options based on loaded data"""
        try:
            if not self.db.is_loaded or not self.db.products:
                return
            
            # Get unique brands
            brands = set()
            
            for product in self.db.products:
                brand = product.get('thuong_hieu', '')
                if brand and brand.strip():
                    brands.add(brand.strip())
            
            # Update brand combobox
            brand_values = ["Tất cả"] + sorted(list(brands))
            self.brand_combo['values'] = brand_values
            
            logger.info(f"Updated filter options: {len(brands)} brands")
            
        except Exception as e:
            logger.error(f"Update filter options error: {str(e)}")
    
    def _create_status_panel(self):
        """Create status bar"""
        try:
            status_frame = tk.Frame(self.root, bg='#34495e', height=30)
            status_frame.pack(fill='x', side='bottom')
            status_frame.pack_propagate(False)
            
            self.status_var = tk.StringVar(value="Sẵn sàng - Vui lòng chọn file JSON")
            status_label = tk.Label(status_frame, 
                                  textvariable=self.status_var, 
                                  bg='#34495e', 
                                  fg='white', 
                                  font=('Arial', 10))
            status_label.pack(side='left', padx=10, pady=5)
            
            # Progress bar
            self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
            self.progress.pack(side='right', padx=10, pady=5)
            
        except Exception as e:
            logger.error(f"Status panel error: {str(e)}")
            raise
    
    def _initialize_database(self):
        """Initialize the database"""
        if not self.json_file:
            self._update_status("Vui lòng chọn file JSON")
            return
            
        try:
            self._update_status("Đang tải và huấn luyện mô hình...")
            self.progress.start()
            
            def init_db():
                try:
                    self.db.load_file(self.json_file)
                    self.root.after(0, self._on_db_initialized)
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda: self._on_db_error(error_msg))
            
            threading.Thread(target=init_db, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Database init error: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize database: {str(e)}")
    
    def _on_db_initialized(self):
        """Database initialized callback"""
        try:
            self.progress.stop()
            self._update_status(f"Đã tải {len(self.db.products)} sản phẩm và huấn luyện hệ thống tìm kiếm 4 bước")
            self._update_filter_options()
            self._display_all_products()
        except Exception as e:
            logger.error(f"Database init callback error: {str(e)}")
    
    def _on_db_error(self, error_msg):
        """Database error callback"""
        self.progress.stop()
        self._update_status("Lỗi tải dữ liệu")
        messagebox.showerror("Database Error", f"Không thể tải file JSON:\n{error_msg}")
    
    def _update_status(self, message):
        """Update status bar"""
        try:
            self.status_var.set(message)
            self.root.update_idletasks()
            logger.info(f"Status: {message}")
        except Exception as e:
            logger.error(f"Status update error: {str(e)}")
    
    def _clear_filters(self):
        """Clear all search filters"""
        try:
            self.search_entry.delete("1.0", tk.END)
            self.price_example_var.set("")
            self.brand_var.set("Tất cả")
            
            self._update_status("Đã xóa bộ lọc")
            if self.db.is_loaded:
                self._display_all_products()
            
        except Exception as e:
            logger.error(f"Clear filters error: {str(e)}")
            messagebox.showerror("Error", f"Failed to clear filters: {str(e)}")
    
    def _retrain_model(self):
        """Retrain the vector model"""
        try:
            if not self.db or not self.db.is_loaded:
                messagebox.showwarning("Warning", "Vui lòng chọn và tải file JSON trước")
                return
            
            self._update_status("Đang huấn luyện lại mô hình...")
            self.progress.start()
            
            def retrain():
                try:
                    self.db.vector_db.train(self.db.products)
                    self.root.after(0, self._on_retrain_completed)
                except Exception as e:
                    self.root.after(0, lambda: self._on_search_error(str(e)))
            
            threading.Thread(target=retrain, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Retrain error: {str(e)}")
            messagebox.showerror("Error", f"Failed to retrain model: {str(e)}")
    
    def _on_retrain_completed(self):
        """Retrain completed callback"""
        try:
            self.progress.stop()
            self._update_status("Hoàn thành huấn luyện lại mô hình")
            messagebox.showinfo("Success", "Mô hình đã được huấn luyện lại thành công!")
        except Exception as e:
            logger.error(f"Retrain completion error: {str(e)}")
    
    def _on_search_error(self, error_msg):
        """Search error callback"""
        self.progress.stop()
        self._update_status("Lỗi tìm kiếm")
        messagebox.showerror("Search Error", f"Search failed:\n{error_msg}")
    
    def _display_all_products(self):
        """Display first 20 products as sample"""
        try:
            if self.db and self.db.is_loaded and self.db.products:
                sample_products = self.db.products[:20]
                self._display_sample_products(sample_products)
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Không có sản phẩm nào để hiển thị")
        except Exception as e:
            logger.error(f"Display all products error: {str(e)}")
    
    def _display_sample_products(self, products):
        """Display sample products"""
        try:
            self.results_text.delete(1.0, tk.END)
            
            self.results_text.insert(tk.END, f"📋 HIỂN THỊ MẪU {len(products)} SẢN PHẨM\n", 'query_info')
            self.results_text.insert(tk.END, f"Tổng số sản phẩm trong database: {len(self.db.products)}\n\n", 'spec')
            self.results_text.insert(tk.END, f"{'='*80}\n\n", 'spec')
            
            for i, product in enumerate(products, 1):
                self.results_text.insert(tk.END, f"🏠 {i}. ", 'spec')
                self.results_text.insert(tk.END, f"{product.get('ten_san_pham', 'N/A')}\n", 'title')
                self.results_text.insert(tk.END, f"   Thương hiệu: {product.get('thuong_hieu', 'N/A')}\n", 'brand')
                self.results_text.insert(tk.END, f"   Giá: {product.get('gia_tien', 'N/A')}\n", 'price')
                self.results_text.insert(tk.END, f"   Công suất: {product.get('cong_suat', 'N/A')}\n", 'spec')
                self.results_text.insert(tk.END, "\n")
            
            self.results_text.see(1.0)
            
        except Exception as e:
            logger.error(f"Display sample products error: {str(e)}")
    
    def run(self):
        """Run the application"""
        try:
            logger.info("Starting 4-Step Smart Search GUI application")
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            messagebox.showerror("Critical Error", f"Application failed:\n{str(e)}")
    
    def _on_closing(self):
        """Handle window closing"""
        try:
            logger.info("Closing application...")
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            logger.error(f"Closing error: {str(e)}")

def main():
    """Main function"""
    try:
        logger.info("🚀 Starting enhanced 4-step search application")
        
        # Check dependencies
        try:
            import sklearn
            import numpy
            import unidecode
        except ImportError as e:
            missing_lib = str(e).split()[-1].replace("'", "")
            messagebox.showerror("Missing Dependencies", 
                               f"Required library missing: {missing_lib}\n\n"
                               "Please install with:\n"
                               f"pip install {missing_lib}")
            return
        
        app = ApplianceSearchGUI()
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        messagebox.showerror("Critical Error", f"Application cannot run:\n{str(e)}")
    finally:
        logger.info("Application ended")

if __name__ == "__main__":
    main()

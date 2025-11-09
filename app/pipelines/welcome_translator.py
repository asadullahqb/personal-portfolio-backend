TRANSLATIONS = {
    "US": "Welcome", "GB": "Welcome", "FR": "Bienvenue",
    "ES": "Bienvenido", "DE": "Willkommen", "CN": "欢迎",
    "JP": "ようこそ", "IN": "स्वागत है", "MY": "Selamat Datang",
    "default": "Welcome"
}

IP_TO_COUNTRY = {
    "192.": "US", "172.": "GB", "10.": "FR", "8.": "DE",
    "14.": "CN", "16.": "JP", "20.": "IN", "25.": "MY"
}

def get_country_from_ip(ip: str) -> str:
    for prefix, country in IP_TO_COUNTRY.items():
        if ip.startswith(prefix):
            return country
    return "default"

def translate_welcome(ip: str) -> str:
    country = get_country_from_ip(ip)
    return TRANSLATIONS.get(country, TRANSLATIONS["default"])

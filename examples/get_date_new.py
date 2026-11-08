import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import os
import sys
import time
import random

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from kronos_safety import safe_named_file


def get_stock_market(stock_code):
    """
    根据股票代码判断市场类型
    返回: 市场前缀 '0'-深交所, '1'-上交所
    """
    if stock_code.startswith(('0', '2', '3')):
        return '0'  # 深交所
    elif stock_code.startswith(('6', '9')):
        return '1'  # 上交所
    else:
        return '1'  # 默认上交所


def get_stock_data_eastmoney_all_history(stock_code="002354"):
    """Market-data fetch is not deployed. No request is sent."""
    from ingestion_gate import refuse_market_ingestion
    refuse_market_ingestion("eastmoney")


def parse_kline_data_directly_all_history(response_text, stock_code):
    """
    直接解析K线数据（当JSON解析失败时使用）- 全历史版本
    """
    try:
        # 尝试直接从响应文本中提取K线数据
        if '"klines":[' in response_text:
            start_idx = response_text.find('"klines":[') + 10
            end_idx = response_text.find(']', start_idx)
            klines_str = response_text[start_idx:end_idx]

            # 清理字符串并分割
            klines = [k.strip().strip('"') for k in klines_str.split('","') if k.strip()]

            stock_data = []
            for kline in klines:
                if kline.strip():
                    items = kline.split(',')
                    if len(items) >= 6:
                        stock_data.append({
                            '日期': items[0],
                            '股票代码': stock_code,
                            '开盘价': float(items[1]),
                            '收盘价': float(items[2]),
                            '最高价': float(items[3]),
                            '最低价': float(items[4]),
                            '成交量': float(items[5]),
                            '成交额': float(items[6]) if len(items) > 6 else 0,
                        })

            if stock_data:
                df = pd.DataFrame(stock_data)
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                df = df.sort_index()
                print(f"✅ 直接解析获取 {len(df)} 条历史数据")
                return df
    except Exception as e:
        print(f"❌ 直接解析也失败: {e}")

    return None


def get_stock_data_akshare_all_history(stock_code="002354"):
    """Market-data fetch is not deployed. No request is sent."""
    from ingestion_gate import refuse_market_ingestion
    refuse_market_ingestion("akshare")


def get_stock_data_baostock_all_history(stock_code="002354"):
    """Market-data fetch is not deployed. No request is sent."""
    from ingestion_gate import refuse_market_ingestion
    refuse_market_ingestion("baostock")


def get_stock_data_with_retry_all_history(stock_code="002354", retry_count=2):
    """
    带重试机制的数据获取 - 多数据源全历史版本
    """
    data_sources = [
        ("AKShare", get_stock_data_akshare_all_history),
        ("Baostock", get_stock_data_baostock_all_history),
        ("东方财富", get_stock_data_eastmoney_all_history)
    ]

    for source_name, data_func in data_sources:
        print(f"\n🔍 尝试从 {source_name} 获取全部历史数据...")
        data = data_func(stock_code)

        if data is not None and not data.empty:
            print(f"✅ {source_name} 历史数据获取成功！")
            # 标记数据来源
            data.attrs['data_source'] = source_name
            return data

    print("❌ 所有真实数据源都失败，使用示例数据...")
    return create_sample_data_all_history(stock_code)


def create_sample_data_all_history(stock_code="002354"):
    """
    创建更真实的历史示例数据 - 从上市年份开始
    """
    # 模拟不同股票的上市年份
    list_years = {
        '600580': 2002,  # 卧龙电驱
        '002354': 2010,  # 天娱数科
        '300418': 2015,  # 昆仑万维
        '300207': 2011,  # 欣旺达
    }

    list_year = list_years.get(stock_code, 2010)
    current_year = datetime.now().year

    print(f"📊 创建 {stock_code} 从 {list_year} 年上市至今的示例数据...")

    # 生成从上市年份到现在的交易日（排除周末）
    start_date = datetime(list_year, 1, 1)
    end_date = datetime.now()
    all_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')

    # 生成更真实的股价数据
    import numpy as np
    np.random.seed(42)

    # 设置合理的基准价格（根据股票类型）
    base_prices = {
        '600580': 8.0,  # 卧龙电驱
        '002354': 15.0,  # 天娱数科 - 上市时价格较高
        '300418': 20.0,  # 昆仑万维
        '300207': 12.0,  # 欣旺达
    }
    base_price = base_prices.get(stock_code, 10.0)

    stock_data = []
    current_price = base_price

    for i, date in enumerate(all_dates):
        # 模拟真实的市场波动
        volatility = 0.02  # 2%的日波动率

        if i > 0:
            # 使用随机游走模拟价格变化
            daily_return = np.random.normal(0, volatility)

            # 模拟不同年份的市场趋势
            year = date.year
            if year <= list_year + 2:  # 上市初期波动较大
                daily_return += np.random.normal(0.001, 0.01)
            elif year <= list_year + 5:  # 成长期
                daily_return += np.random.normal(0.0005, 0.005)
            else:  # 成熟期
                daily_return += np.random.normal(0.0002, 0.003)

            current_price = current_price * (1 + daily_return)

            # 价格边界限制
            current_price = max(base_price * 0.3, min(base_price * 10.0, current_price))
        else:
            current_price = base_price

        # 生成OHLC数据
        open_variation = np.random.normal(0, volatility * 0.2)
        open_price = current_price * (1 + open_variation)

        daily_range = abs(np.random.normal(volatility * 0.8, volatility * 0.3))
        high_price = max(open_price, current_price) * (1 + daily_range)
        low_price = min(open_price, current_price) * (1 - daily_range)
        close_price = current_price

        # 确保价格合理性
        high_price = max(open_price, close_price, low_price, high_price)
        low_price = min(open_price, close_price, high_price, low_price)

        # 生成成交量（随年份增长）
        base_volume = 100000 + (year - list_year) * 50000  # 成交量逐年增长
        volume_variation = abs(daily_return) * 5000000 if i > 0 else 0
        volume = int(base_volume + volume_variation + np.random.randint(-200000, 400000))
        volume = max(50000, volume)

        # 计算成交额（万元）
        amount = volume * close_price / 10000

        # 计算涨跌幅和涨跌额
        if i > 0:
            prev_close = stock_data[-1]['收盘价']
            price_change = close_price - prev_close
            pct_change = (price_change / prev_close) * 100
        else:
            price_change = 0
            pct_change = 0

        # 计算振幅
        amplitude = ((high_price - low_price) / open_price) * 100

        # 生成换手率（1%-15%之间）
        turnover_rate = np.random.uniform(1.0, 15.0)

        stock_data.append({
            '日期': date,
            '股票代码': stock_code,
            '开盘价': round(open_price, 2),
            '收盘价': round(close_price, 2),
            '最高价': round(high_price, 2),
            '最低价': round(low_price, 2),
            '成交量': volume,
            '成交额': round(amount, 2),
            '振幅': round(amplitude, 2),
            '涨跌幅': round(pct_change, 2),
            '涨跌额': round(price_change, 2),
            '换手率': round(turnover_rate, 2)
        })

    df = pd.DataFrame(stock_data)
    df.set_index('日期', inplace=True)

    print(f"✅ 已创建 {len(df)} 条从 {list_year} 年至今的模拟历史数据")
    print(f"时间范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")

    # 标记为模拟数据
    df.attrs['data_source'] = '模拟历史数据'

    return df


def display_all_history_data_info(df, stock_code):
    """显示全历史数据信息"""
    if df is None or df.empty:
        print("没有数据可显示")
        return

    # 获取数据来源
    data_source = df.attrs.get('data_source', '未知来源')

    print(f"\n{'=' * 60}")
    print(f"股票 {stock_code} 全部历史数据摘要")
    print(f"{'=' * 60}")

    print(f"数据时间范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
    print(f"总交易天数: {len(df):,}")
    print(f"数据来源: {data_source}")

    # 按年份显示统计
    years = sorted(df.index.year.unique())
    print(f"\n历史年份: {years}")

    # 显示关键年份统计
    key_years = [years[0]]  # 上市年份
    if len(years) > 1:
        key_years.append(years[-1])  # 最新年份
    if len(years) > 5:
        key_years.extend([years[len(years) // 2], years[len(years) // 4], years[3 * len(years) // 4]])

    for year in sorted(set(key_years)):
        year_data = df[df.index.year == year]
        if len(year_data) > 0:
            print(f"\n{year}年统计:")
            print(f"  交易天数: {len(year_data)}")
            print(f"  平均收盘价: {year_data['收盘价'].mean():.2f} 元")
            print(f"  最高价: {year_data['最高价'].max():.2f} 元")
            print(f"  最低价: {year_data['最低价'].min():.2f} 元")
            if len(year_data) > 1:
                year_return = (year_data['收盘价'].iloc[-1] / year_data['收盘价'].iloc[0] - 1) * 100
                print(f"  年度涨跌幅: {year_return:+.2f}%")

    # 显示整体统计
    print(f"\n整体统计:")
    total_return = (df['收盘价'].iloc[-1] / df['收盘价'].iloc[0] - 1) * 100
    print(f"  总涨跌幅: {total_return:+.2f}%")
    print(f"  历史最高价: {df['最高价'].max():.2f} 元")
    print(f"  历史最低价: {df['最低价'].min():.2f} 元")
    print(f"  平均日成交量: {df['成交量'].mean():,.0f} 股")

    # 显示最新交易日数据
    latest_date = df.index.max()
    print(f"\n最新交易日 ({latest_date.strftime('%Y-%m-%d')}) 数据:")
    latest_data = df.loc[latest_date]
    for col, value in latest_data.items():
        if col != '股票代码':
            if col in ['成交量']:
                print(f"  {col}: {value:,.0f}")
            elif col in ['成交额']:
                print(f"  {col}: {value:,.2f} 万元")
            else:
                print(f"  {col}: {value}")


def save_all_history_stock_data(df, stock_code, save_dir="D:/lianghuajiaoyi/Kronos/examples/data"):
    """
    保存全历史股票数据到指定目录
    """
    if df is not None and not df.empty:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 保存CSV文件 - 使用全历史命名
        csv_file = str(safe_named_file(save_dir, stock_code, "", "_all_history.csv"))

        # 重置索引以便保存日期列
        df_reset = df.reset_index()
        df_reset.to_csv(csv_file, encoding='utf-8-sig', index=False)

        print(f"\n📁 全历史股票数据已保存: {csv_file}")

        # 同时保存一个按年份分割的版本
        years = df_reset['日期'].dt.year.unique()
        for year in years:
            year_data = df_reset[df_reset['日期'].dt.year == year]
            year_file = str(safe_named_file(save_dir, stock_code, "", f"_{int(year)}.csv"))
            year_data.to_csv(year_file, encoding='utf-8-sig', index=False)

        print(f"📁 同时保存了 {len(years)} 个年份的单独数据文件")
        return True
    return False


def main_all_history(stock_code="002354"):
    """
    主函数：获取并保存股票全历史数据
    """
    # 设置保存目录
    save_directory = "D:/lianghuajiaoyi/Kronos/examples/data"

    print("=" * 60)
    print(f"开始获取股票 {stock_code} 的全部历史数据")
    print("=" * 60)
    print(f"数据将保存到: {save_directory}")

    # 检查必要库
    try:
        import requests
        import numpy as np
    except ImportError:
        print("Missing requests or numpy. Install pinned dependencies from requirements.txt and retry.")
        return

    # 获取全历史数据（多数据源）
    stock_data = get_stock_data_with_retry_all_history(stock_code)

    if stock_data is not None:
        # 显示数据信息
        display_all_history_data_info(stock_data, stock_code)

        # 保存全历史数据到指定目录
        save_all_history_stock_data(stock_data, stock_code, save_directory)

        print(f"\n🎉 股票 {stock_code} 全历史数据处理完成!")
        print(
            f"数据时间跨度: {stock_data.index.min().strftime('%Y-%m-%d')} 到 {stock_data.index.max().strftime('%Y-%m-%d')}")
        print(f"总交易天数: {len(stock_data):,}")

        # 显示保存的文件
        csv_file = str(safe_named_file(save_directory, stock_code, "", "_all_history.csv"))
        if os.path.exists(csv_file):
            file_size = os.path.getsize(csv_file) / 1024  # KB
            print(f"📄 生成的文件: {csv_file} ({file_size:.1f} KB)")
    else:
        print("❌ 未能获取股票全历史数据")


# 使用方法说明
if __name__ == "__main__":
    """
    使用方法：
    修改下面的参数来获取不同股票的全历史数据
    """

    # ==================== 在这里修改参数 ====================
    TARGET_STOCK_CODE = "300418"  # 股票代码
    # =====================================================

    print("股票全历史数据获取工具")
    print("说明：修改代码中的 TARGET_STOCK_CODE 来获取不同股票的全部历史数据")
    print(f"当前设置: 股票代码={TARGET_STOCK_CODE}")
    print()

    # 运行主程序
    main_all_history(stock_code=TARGET_STOCK_CODE)

    print(f"\n💡 提示：要获取其他股票的全历史数据，请修改代码中的 TARGET_STOCK_CODE 变量")
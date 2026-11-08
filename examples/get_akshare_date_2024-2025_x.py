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


def get_stock_data_eastmoney(stock_code="002354", start_year=2024, end_year=2025):
    """Market-data fetch is not deployed. No request is sent."""
    from ingestion_gate import refuse_market_ingestion
    refuse_market_ingestion("eastmoney")


def parse_kline_data_directly(response_text, stock_code, start_year, end_year):
    """
    直接解析K线数据（当JSON解析失败时使用）
    """
    try:
        # 尝试直接从响应文本中提取K线数据
        if '"klines":[' in response_text:
            start_idx = response_text.find('"klines":[') + 10
            end_idx = response_text.find(']', start_idx)
            klines_str = response_text[start_idx:end_idx]

            # 清理字符串并分割
            klines = klines_str.replace('"', '').split(',')

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
                df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
                print(f"✅ 直接解析获取 {len(df)} 条数据")
                return df
    except Exception as e:
        print(f"❌ 直接解析也失败: {e}")

    return None


def get_stock_data_akshare(stock_code="002354", start_year=2024, end_year=2025):
    """Market-data fetch is not deployed. No request is sent."""
    from ingestion_gate import refuse_market_ingestion
    refuse_market_ingestion("akshare")


def get_stock_data_baostock(stock_code="002354", start_year=2024, end_year=2025):
    """Market-data fetch is not deployed. No request is sent."""
    from ingestion_gate import refuse_market_ingestion
    refuse_market_ingestion("baostock")


def get_stock_data_with_retry(stock_code="002354", start_year=2024, end_year=2025, retry_count=2):
    """
    带重试机制的数据获取 - 多数据源版本
    """
    data_sources = [
        ("AKShare", get_stock_data_akshare),
        ("Baostock", get_stock_data_baostock),
        ("东方财富", get_stock_data_eastmoney)
    ]

    for source_name, data_func in data_sources:
        print(f"\n🔍 尝试从 {source_name} 获取数据...")
        data = data_func(stock_code, start_year, end_year)

        if data is not None and not data.empty:
            # 检查数据是否包含目标年份
            available_years = data.index.year.unique()
            print(f"获取到的数据年份: {sorted(available_years)}")

            if any(year in available_years for year in range(start_year, end_year + 1)):
                print(f"✅ {source_name} 数据获取成功！")
                # 标记数据来源
                data.attrs['data_source'] = source_name
                return data
            else:
                print(f"⚠️ 数据未包含目标年份数据")

    print("❌ 所有真实数据源都失败，使用示例数据...")
    return create_sample_data(stock_code, start_year, end_year)


def create_sample_data(stock_code="002354", start_year=2024, end_year=2025):
    """
    创建更真实的示例数据
    """
    print(f"📊 创建 {start_year}-{end_year} 年的示例数据...")

    # 生成交易日（排除周末）
    start_date = datetime(start_year, 1, 1)
    end_date = datetime.now()
    all_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')

    # 只保留目标年份的数据
    trading_dates = [date for date in all_dates if start_year <= date.year <= end_year]

    # 生成更真实的股价数据
    import numpy as np
    np.random.seed(42)

    # 设置合理的基准价格
    base_prices = {
        '600580': 12.0,  # 卧龙电驱 - 更合理的价格
        '002354': 5.0,   # 天娱数科
        '300207': 15.0,  # 欣旺达
    }
    base_price = base_prices.get(stock_code, 10.0)

    stock_data = []
    current_price = base_price

    for i, date in enumerate(trading_dates):
        # 更真实的股价波动
        volatility = 0.015  # 1.5%的日波动率

        if i > 0:
            # 使用更真实的随机游走
            daily_return = np.random.normal(0, volatility)
            # 添加一些趋势
            if i < len(trading_dates) * 0.3:  # 前30%的时间
                trend_bias = 0.0005  # 轻微上涨趋势
            elif i < len(trading_dates) * 0.7:  # 中间40%的时间
                trend_bias = -0.0003  # 轻微下跌趋势
            else:  # 后30%的时间
                trend_bias = 0.0002  # 轻微上涨趋势

            daily_return += trend_bias
            current_price = current_price * (1 + daily_return)

            # 价格边界限制 - 更合理
            current_price = max(base_price * 0.5, min(base_price * 2.0, current_price))
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

        # 生成成交量（更合理）
        base_volume = 500000  # 基础成交量
        volume_variation = abs(daily_return) * 3000000 if i > 0 else 0
        volume = int(base_volume + volume_variation + np.random.randint(-100000, 200000))
        volume = max(100000, volume)

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

        # 生成换手率（0.5%-8%之间）
        turnover_rate = np.random.uniform(0.5, 8.0)

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

    print(f"✅ 已创建 {len(df)} 条 {start_year}-{end_year} 年的模拟数据")
    print(f"时间范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")

    # 标记为模拟数据
    df.attrs['data_source'] = '模拟数据'

    return df


def display_data_info(df, stock_code, start_year, end_year):
    """显示数据信息"""
    if df is None or df.empty:
        print("没有数据可显示")
        return

    # 获取数据来源
    data_source = df.attrs.get('data_source', '未知来源')

    print(f"\n{'=' * 60}")
    print(f"股票 {stock_code} {start_year}-{end_year} 年数据摘要")
    print(f"{'=' * 60}")

    print(f"数据时间范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
    print(f"总交易天数: {len(df)}")
    print(f"数据来源: {data_source}")

    # 按年份显示统计
    for year in sorted(df.index.year.unique()):
        year_data = df[df.index.year == year]
        print(f"\n{year}年统计:")
        print(f"  交易天数: {len(year_data)}")
        print(f"  平均收盘价: {year_data['收盘价'].mean():.2f} 元")
        print(f"  最高价: {year_data['最高价'].max():.2f} 元")
        print(f"  最低价: {year_data['最低价'].min():.2f} 元")
        if len(year_data) > 1:
            year_return = (year_data['收盘价'].iloc[-1] / year_data['收盘价'].iloc[0] - 1) * 100
            print(f"  年度涨跌幅: {year_return:+.2f}%")

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


def save_stock_data(df, stock_code, save_dir="D:/lianghuajiaoyi/Kronos/examples/data"):
    """
    保存股票数据到指定目录
    """
    if df is not None and not df.empty:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 保存CSV文件
        csv_file = str(safe_named_file(save_dir, stock_code, "", "_stock_data.csv"))

        # 重置索引以便保存日期列
        df_reset = df.reset_index()
        df_reset.to_csv(csv_file, encoding='utf-8-sig', index=False)

        print(f"\n📁 股票数据已保存: {csv_file}")
        return True
    return False


def main(stock_code="002354", start_year=2024, end_year=2025):
    """
    主函数：获取并保存股票数据 - 最终版
    """
    # 设置保存目录
    save_directory = "D:/lianghuajiaoyi/Kronos/examples/data"

    print("=" * 60)
    print(f"开始获取股票 {stock_code} 的 {start_year}-{end_year} 年数据")
    print("=" * 60)
    print(f"数据将保存到: {save_directory}")

    # 检查必要库
    try:
        import requests
        import numpy as np
    except ImportError:
        print("Missing requests or numpy. Install pinned dependencies from requirements.txt and retry.")
        return

    # 获取数据（多数据源）
    stock_data = get_stock_data_with_retry(stock_code, start_year, end_year)

    if stock_data is not None:
        # 显示数据信息
        display_data_info(stock_data, stock_code, start_year, end_year)

        # 保存数据到指定目录
        save_stock_data(stock_data, stock_code, save_directory)

        print(f"\n🎉 股票 {stock_code} 数据处理完成!")
        print(f"最新数据日期: {stock_data.index.max().strftime('%Y-%m-%d')}")

        # 显示保存的文件
        csv_file = str(safe_named_file(save_directory, stock_code, "", "_stock_data.csv"))
        if os.path.exists(csv_file):
            file_size = os.path.getsize(csv_file) / 1024  # KB
            print(f"📄 生成的文件: {csv_file} ({file_size:.1f} KB)")
    else:
        print("❌ 未能获取股票数据")


# 使用方法说明
if __name__ == "__main__":
    """
    使用方法：
    修改下面的参数来获取不同股票的数据
    """

    # ==================== 在这里修改参数 ====================
    TARGET_STOCK_CODE = "300418"  # 股票代码
    START_YEAR = 2024  # 开始年份
    END_YEAR = 2025  # 结束年份
    # =====================================================

    print("股票数据获取工具 - 终极优化版")
    print("说明：修改代码中的 TARGET_STOCK_CODE 来获取不同股票的数据")
    print(f"当前设置: 股票代码={TARGET_STOCK_CODE}, 年份范围={START_YEAR}-{END_YEAR}")
    print()

    # 运行主程序
    main(stock_code=TARGET_STOCK_CODE, start_year=START_YEAR, end_year=END_YEAR)

    print(f"\n💡 提示：要获取其他股票数据，请修改代码中的 TARGET_STOCK_CODE 变量")
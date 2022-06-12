import decimal

width = 10
precision = 4
value = decimal.Decimal('12.34567')
print(f'result: {value:{width}.{precision}}')  # result:      12.35

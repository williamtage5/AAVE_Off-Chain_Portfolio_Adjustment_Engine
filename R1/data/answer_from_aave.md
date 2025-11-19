{
  reserves(where: {isActive: true}, first: 100) {
    symbol
    decimals
    totalLiquidity
    totalLiquidityAsCollateral
    totalPrincipalStableDebt
    totalCurrentVariableDebt
    reserveLiquidationThreshold
    price {
      priceInEth
      id
    }
  }
}

{
  "data": {
    "reserves": [
      {
        "decimals": 18,
        "price": {
          "id": "0x04c0599ae5a44757c0af6f9ec3b93da8976c150a",
          "priceInEth": "0"
        },
        "reserveLiquidationThreshold": "7700",
        "symbol": "weETH",
        "totalCurrentVariableDebt": "5711413226611483813",
        "totalLiquidity": "101060475476073517363523",
        "totalLiquidityAsCollateral": "80304996875868344605137",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 18,
        "price": {
          "id": "0x236aa50979d5f3de3bd1eeb40e81137f22ab794b",
          "priceInEth": "9691461317600"
        },
        "reserveLiquidationThreshold": "7800",
        "symbol": "tBTC",
        "totalCurrentVariableDebt": "2319486623176828625",
        "totalLiquidity": "10340642374716541723",
        "totalLiquidityAsCollateral": "-27524976671639834275",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 18,
        "price": {
          "id": "0x2416092f143378750bb29b79ed961ab195cceea5",
          "priceInEth": "0"
        },
        "reserveLiquidationThreshold": "10",
        "symbol": "ezETH",
        "totalCurrentVariableDebt": "0",
        "totalLiquidity": "62724757656228962796",
        "totalLiquidityAsCollateral": "-200445018176130775600",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 18,
        "price": {
          "id": "0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22",
          "priceInEth": "0"
        },
        "reserveLiquidationThreshold": "7900",
        "symbol": "cbETH",
        "totalCurrentVariableDebt": "73211718930522736606",
        "totalLiquidity": "7275725911966508937708",
        "totalLiquidityAsCollateral": "-7169221468361580822183",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 18,
        "price": {
          "id": "0x4200000000000000000000000000000000000006",
          "priceInEth": "322840000000"
        },
        "reserveLiquidationThreshold": "8300",
        "symbol": "WETH",
        "totalCurrentVariableDebt": "111935033899531092134493",
        "totalLiquidity": "135773819228238313273787",
        "totalLiquidityAsCollateral": "527248943483862701017704",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 6,
        "price": {
          "id": "0x60a3e35cc302bfa44cb288bc5a4f316fdb1adb42",
          "priceInEth": "0"
        },
        "reserveLiquidationThreshold": "7800",
        "symbol": "EURC",
        "totalCurrentVariableDebt": "14738394184678",
        "totalLiquidity": "28137923993177",
        "totalLiquidityAsCollateral": "-26553894922173",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 18,
        "price": {
          "id": "0x63706e401c06ac8513145b7687a14804d17f814b",
          "priceInEth": "18186543000"
        },
        "reserveLiquidationThreshold": "6500",
        "symbol": "AAVE",
        "totalCurrentVariableDebt": "0",
        "totalLiquidity": "5743469788964741943928",
        "totalLiquidityAsCollateral": "-32973120490976376246439",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 18,
        "price": {
          "id": "0x6bb7a212910682dcfdbd5bcbb3e28fb4e8da10ee",
          "priceInEth": "0"
        },
        "reserveLiquidationThreshold": "0",
        "symbol": "GHO",
        "totalCurrentVariableDebt": "7831765623389494312280982",
        "totalLiquidity": "14562490328570051552693819",
        "totalLiquidityAsCollateral": "0",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 6,
        "price": {
          "id": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
          "priceInEth": "99986468"
        },
        "reserveLiquidationThreshold": "7800",
        "symbol": "USDC",
        "totalCurrentVariableDebt": "270641188114918",
        "totalLiquidity": "394263161444332",
        "totalLiquidityAsCollateral": "-992527629617345",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 18,
        "price": {
          "id": "0xc1cba3fcea344f92d9239c08c0568f6f2f0ee452",
          "priceInEth": "0"
        },
        "reserveLiquidationThreshold": "7900",
        "symbol": "wstETH",
        "totalCurrentVariableDebt": "6871703749728494496329",
        "totalLiquidity": "27527287398506498472580",
        "totalLiquidityAsCollateral": "8793409556879785399454",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 8,
        "price": {
          "id": "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf",
          "priceInEth": "8155695500000"
        },
        "reserveLiquidationThreshold": "7800",
        "symbol": "cbBTC",
        "totalCurrentVariableDebt": "40231161143",
        "totalLiquidity": "253922858089",
        "totalLiquidityAsCollateral": "-750057277287",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 6,
        "price": {
          "id": "0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca",
          "priceInEth": "100010000"
        },
        "reserveLiquidationThreshold": "7800",
        "symbol": "USDbC",
        "totalCurrentVariableDebt": "403423125522",
        "totalLiquidity": "437671384568",
        "totalLiquidityAsCollateral": "36087202925214",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 8,
        "price": {
          "id": "0xecac9c5f704e954931349da37f60e39f515c11c1",
          "priceInEth": "10849218252036"
        },
        "reserveLiquidationThreshold": "7300",
        "symbol": "LBTC",
        "totalCurrentVariableDebt": "0",
        "totalLiquidity": "5859006953",
        "totalLiquidityAsCollateral": "-1563923280",
        "totalPrincipalStableDebt": "0"
      },
      {
        "decimals": 18,
        "price": {
          "id": "0xedfa23602d0ec14714057867a78d01e94176bea0",
          "priceInEth": "0"
        },
        "reserveLiquidationThreshold": "10",
        "symbol": "wrsETH",
        "totalCurrentVariableDebt": "0",
        "totalLiquidity": "6478564545149594243854",
        "totalLiquidityAsCollateral": "3704794255875495187900",
        "totalPrincipalStableDebt": "0"
      }
    ]
  }
}

# import os
# import tqdm
# import json
# from visual_nuscenes import NuScenes
# use_gt = False
# out_dir = '/home/docker_rctrans/RCTrans/result_vis/origin_cars/'
# result_json = "/home/docker_rctrans/RCTrans/results/origin_results_nusc"
# dataroot='/home/docker_rctrans/HPR3/nuscenes'

# os.makedirs(out_dir, exist_ok=True)

# if use_gt:
#     nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
# else:
#     nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

# with open('{}.json'.format(result_json)) as f:
#     table = json.load(f)
# tokens = list(table['results'].keys())
# index=0
# cars_tokens = ['381fe530586b4c189e10cbefbfb0e773',
#  'f0e4797551024f9487a016ee9d9e29e1',
#  '72f030bd5dd142d497c090dc04bb5697',
#  'dbffcc96729b420380ace0e08ad587d5',
#  'cf5c4b437daf4bb3ad68d0f88ea7bff2',
#  '561cf9acfda645b79bc52ec82537970c',
#  'fd534bf0c87d4d4cb891f0fc6f41a6a9',
#  'ff98be19bcf9436cad14f28ff63cb501',
#  '4b6fcdc1549847c395d07fefcbdc16eb',
#  'b1eea1e3d59e4afdaa6d29481480d974',
#  '05cf2a68a2cf4ff1838034f8ecae0c82',
#  'b04b95006e24460bacaaf80e299dff01',
#  'b006ca82a2654951ae8f42300f66f3c4',
#  '3307d1211dfd442db4a6ae544c2ae197',
#  '50d86a5cd45440c89e895a5d20ea5267',
#  'fe944294f6a44b87a4da95136f86bb79',
#  'a9b8176bf0b546a4bc46afc631979805',
#  '0e8782aa721545caabc7073d32fb1fb1',
#  '8d1c7b2b85fe4befb66efe9e05e0d939',
#  'd22de14af8cc4007994efda8354c939b',
#  '5f88b9a41f264ebbac7976bf4c796d21',
#  '6be3fbcf5c224508adb27b4b703b6625',
#  '9e9808fe898648e1b0bbf09b1a30f0d4',
#  'ae3d3b5cb4424e7ab501a5d6f0a4980a',
#  '88a6b5e35eb44deba6b1065d74b0a3c4',
#  'a687809bdfff490198d3b39bfc4bd42b',
#  '0a0d6b8c2e884134a3b48df43d54c36a',
#  '9e7683e8586542a1b6032980c45f15ce',
#  'b9ea04a6121d4a8bb00199b885aa5ef0',
#  '08ec4b7fcdb5494c8da174770c1d9245',
#  '49823b270a62468397f3265172dcc6bf',
#  '4f2578e3107f4ad79f88a61b28893916',
#  'f3491dccffc3428aac243872cbfe5072',
#  '135bf33890ba4ca2984a931444923eda',
#  '5d773ca713f54023b34cd4718a5ee293',
#  '20d02a3110fb47348e175a61ad157875',
#  '0490bd92372a4a2d98c7136ba6ebcfce',
#  '08d95280dd2f42d98a4a6e33dff8e815',
#  '4f9ad42bb4a24970b770ba0a87baf47a',
#  '33c965ece87842c2ac898543d26dad5f',
#  'e32db3f8b6c244c2be60b6d7db30ccc3',
#  'f38ef51177c84c2a9fa32584b79e601e',
#  'c62b6ac878934607be0c524a02f1692b',
#  '6c662b1258e34752be332dcc1eccb7a8',
#  '5c7c23122dd94c70b8ab5528f27ba117',
#  '7461eb9987354c5db09f9203786b2324',
#  '4fc9bbe6198c4b8e837fd2ae0de6a4e2',
#  'c48c3dfec8b5454ab97303a00365a77b',
#  'c7b18ce7027c4cc3b2068fbf46860b3f',
#  '94126983bc0c4de89fb27cefb81f24ef',
#  'dafe7793990d41a0ae1e50260899f8b4',
#  'c900549f484c4c03872e35d4d927f5d7',
#  '868280bdd9bf4ec49d2dd6d4880d4bee',
#  'e84d37c300094d51aa2917b4ac006da5',
#  'cf2601d5aaee4ab8a700029a33d49be6',
#  '4bc57fd309b8495fa222abc3263b47e5',
#  '7e034567ff9e4c22819fd59c4cb845da',
#  'fc2d1d2139ae40a3b781f70004a8fba8',
#  'a7831d4d1db54053a501d0418545fee2',
#  '8b57f73177694aee8d393b945df0cd38',
#  '2c253adbce79406c962b9382236768da',
#  '5b29d1a2b0e646368622dee5e8fa5108',
#  '1e82fd0644c142a2be36770ee29815aa',
#  '0ad5d20390ec43f59d210c1a3a11b23d',
#  'd7bea9a3552b40e29d42324108dff575',
#  'a572dd2e95e94e4db66bb5dbfaf870b5',
#  'c143943fd6e246448eab88ce1a0aeda9',
#  '8c4d447e628c4393839187bcfa0cbd76']

# night_tokens = ['cb4504edb87a40d4b650cd5860f6c3b7',
#  'a945d7d02c7a4ef98f1b5aaa1557f546',
#  'eb82a835419149aeae82c5e8ea64377b',
#  '49e0e239c3f048418bb250ef9376120e',
#  'e74b8ce0b1824e0c8167f2534d8d4f7d',
#  '2f51a63b389c4a42afa5b546e4f90166',
#  'e0e5ebc17c73465181cab1383606e5f5',
#  '5510e4b6d7f4430fbcc77cf660d9fbb6',
#  '79f6489272c24d3ebc5e225ce6ff2aea',
#  'f8de8186b0b046418127ad7007a206c7',
#  'bb4f3cfb564f47a3bb9b9313e19365c2',
#  'a08ae80f74dd46e0afcb3c740ff6ea53',
#  '28eb331d407e41e0804cdd59b02fb7a4',
#  'e1b741c0b3a342e789365174a36764f6',
#  '17d12b9725ce49fba738ae3a0bb3ca78',
#  '1a776100d0974c0b9edc9bb80a842b54',
#  '4f371688b8af4d2bac93df275036b909',
#  'b615d1afcff44bbab1a3e34f76a64942',
#  '23ef1630fb3c42a7b149d88f162d6a78',
#  'b8849fcd11d14e499aea2ee258d8b581',
#  '4b36703115d547dc8611741eb213da76',
#  'c82417563e44430e9e59224e36872f4e',
#  'cce6922361124fd0b4c5f28acf78d65d',
#  '1fcc8d01ea7048a9a9eb0eb161778026',
#  '9435f31153e845c59b8481126eadc094',
#  '99a4c956cb4f44cc975680994c6bb40a',
#  'a36ddb26940e4863bc8df270c2678b57',
#  '84324196e7234066973546a167720e54',
#  '887e6f18b1b341d9b55bc4289c2e0388',
#  '6e1f02cb4e2747c6970b2d9b2e834852',
#  'c30d133fc18d476e80f8df6e63ce5dc4',
#  'c6749814a54341b19d882ef6fbd5cf86',
#  'ff729b4a16a74350a0faa2b094ebfb2a',
#  '7b0c6e14258946f9aa3b81f39a5e0816',
#  '8418f22ec0d94a08ae038768abae743a',
#  'fcec74d888d745c0964b785dc309cf2c',
#  'f22f8f2fc7314b028419d8332c03a049',
#  'ae8a5000d89c4c998a5d6bf95ba2ace9',
#  'f161848f032840119785996d26004c77',
#  '24dc629e9bf74231affdfc033c79f8d4',
#  '5ba9f3b30a5d42aeb790bfca7648bf77',
#  '11a46849ff264ac3b75ba0709c4a0a35',
#  'bc33987f6a8b47709aa70846e6648877',
#  'f9c8bf233bd946ba8425f97ef3ccc5ce',
#  '3c8e6743e1ed42139ef8e5fbe3a9be17',
#  '79e28a35e07b444c893111137aea81e7',
#  'e0a7e1df190e48feb416dad36d140c35',
#  'a04be2dd20eb493daa336a299f052426',
#  '6a147ca886954584aad7eb68b15db56d',
#  'df957c9b8e194e7f88b9a3dc84f12174',
#  '3d7733347cc04bfa97a43b4de2e6437f',
#  'c3acd43e79464a24b0fadbe1e5ba9950',
#  '5622886f3e7e427fbcb0243cf664afe9',
#  'de18907d85164692bafbb4916484b9b9',
#  '4dad239b9fc048f7a6e69e2ba45b35b1',
#  '0657a919d58a462c8905f7a0c706dafb',
#  'dd873a094c5b47649b79c3f9dc36dfb2']

# rain_tokens = ['67d2d74087714e4994a68c7347acf55a',
#  '98fc274ca14d44a5b9117fbfb04d18b1',
#  '668349dd12da4e1fa5d777634af88c32',
#  '6fde33ada8b04da9929057fd6b85a72b',
#  '57f23d2f55e6455696c6d3cc70a4f501',
#  'a35e04cb7b8f466ca0d4031b483e0b0e',
#  '7e3b2e211788438c88a1840f5d2e6410',
#  '945de6d3e88940269fbfe07ec1feafac',
#  '0fae8aae30d44b4faf7a9854eac4bc09',
#  '0775bcc755c148ef831dffe278b1bb94',
#  '221c593dff9e429cb5e15139cc829c1c',
#  'c8f0119b92774ec68db5e1509888fe92',
#  'd5a661d8c39e4d87bc3d367063a8c82a',
#  'd4b3a58594d848d49dbef38e2e6e2125',
#  'bdbc78adca87461d93ce8b90f1f38a1e',
#  'dedec80061fd4640be7a8945c9846ff6',
#  'a15bd188376049479320d79c39611040',
#  'cfabe989031147e9bed9de9dacd60afa',
#  '72e83c660c194fc988a368cbbb928cac',
#  '64701483680a4d8eae3c8276b611d8d5',
#  '6cefb90da1b1468795b6a3a333c94b38',
#  '20fdb20b4ca148b9bca502ca7c196190',
#  '2e3374eb19ee46e09fd0fa90281fbc7d',
#  'ea13e21a23d04f1a885f67744421ae52',
#  '27e105e2c42b46479433c2d8df488f1d',
#  'fdd22a4e963c48fe8987afff20c46648',
#  '885d559ea0bb4c75ba6be2ad572f068f',
#  '10cb11d8a5554c2a9741b4f9edd88c86',
#  'a0b6daacdda84ad8a2696593b4ef2e4a',
#  'e49931f3efaf4066b5398d2bebf8081b',
#  '6df1e1eb5dff4b019ed0c8b5e9ac14ec',
#  '4e9c011236e64ac3876e8ae12290b881',
#  'bfe534d72d984d99be6d8f40f7e53085',
#  '03e4d13ede7346b6a0f2d602e739795d',
#  'a2a6245f58fb4db897b3514a836f0ef6',
#  'f6c4cb03fbb44ccfb19d19fae99480c8',
#  '7e647f43cd284d669fde775b5b69d2d8',
#  'db29741bf3e2411e953597309225f41c',
#  '0cd5782c481c4ef0b4bce413089f04c0',
#  '8ba8c0aca3e840cdbd68984c998b1adc',
#  'b94fdaf044e2474783bf0d7d18f77f3c',
#  '6f91a4053fe841faab7a91a526ab9e66',
#  '7604362a89f94113ad7a2bb19a074520',
#  'f542a7d55b5c4e97bf42eee1f442356c',
#  '1a2db86db8fb409abc7e26578b966d6e',
#  'a62913cc73e94f5ca721f41fda89d3ad',
#  'a06bcad997144dcb82ca925d698c11cd',
#  'aca5ebf7f3c940dd91c2fab7d0e0f2f2',
#  '904a50cc45a143fb9b3fb1e9889550a6',
#  'c2a4c914bd064ede98dc6b3c193050a8',
#  '95f21212afec4736b0abd977b01d5ded',
#  '5928500817c747ce956c765df7b49283',
#  'aeb18e6a17ef44ab87dcfae207151e02',
#  'a474fc3cae054873a4db787ed25e497a',
#  '7fafcee5a8d1453397d168c3f9568564',
#  'b21b443198bc4552ba7593a471abda1f',
#  '73550331af2e48fd8679ad97178ff356',
#  'ce93f225efbd43ac99b94e885de1b09c',
#  '86a69236763d42bf9da6247b8bcd88bb',
#  'ed10780136f34011b35e1aaa1d8fb1e5',
#  '30507c7761c048c896dc0185e855fd62',
#  '3de1d789fd004720bcef0a42d47c1159',
#  '7b173aed71f84d58bd218628ea584190',
#  '55f7c2992b67437e8ebd21f014b0f345',
#  '7000c2ccd142491bb54b14e361b9b412',
#  '0084b0b7c69f491eb75ed43ac19a47a3',
#  '352c344ba8b045fe9d546d8fc1919fe2',
#  '1a175086f0e248538b0eb83d830dd89a',
#  '60f5b50bd3a146c08d016cee80637f62',
#  '525a91b18a1f4ec284dd244678b47f35',
#  '8119ac63b62b420bb982b7e064168349',
#  '302c95923235437d8d87e8d741e34279',
#  '0562be5d8f284bd38d811469f02c857f',
#  '45df585aa5fb4972986093b353bae03f',
#  '7484388789fe41ee94aaf68011b543ca',
#  '2ec05dce9faf45d39cdf8188471e9ea7',
#  '098e97621e6b4561a58a74a8bc1a91fd',
#  '56a6c2493a4544e3a1f0aaf67db1258d',
#  '2f1fd75cc1f143509dfd4982738da9fe',
#  '83d3ee0e085b4ac282e06741fe1f3ae2',
#  '01653ee2b605497d95848f3e6cac5f67',
#  '1e6f6e90ec654248926d03aaa42a120d',
#  '0071536e4d774034a135085c3f08890e',
#  'ce9df8d818354073b48c2b0e5e8ff7b5',
#  'e60fea7246b54e949468bef871cf5cac',
#  '10e6d20748d4434986cfff5b74222781',
#  '614cb31279b84b329865c216024711dc',
#  'e0513430edd04b22af02eb1d912bfe83',
#  '98605ed784f7455bb52d1994cdbaaaac',
#  'd2f45becb0af4a458e678d21f2b9add2',
#  'bb963264d4ef4e6fa80d4f10bb8b2dfa',
#  '35afe50d131847739f3221c6ffb88d42',
#  '61421b8cc155473c8f1db68c24ebc618',
#  '6350cb22be8c458792dee63332f31617',
#  'ccc19df4b0784fb28d67f76645641bc7',
#  '3adfe1171bb448f9a639e4bc3bc3dc11',
#  '935ddd79f2cc4b6a81395cde46290fff']
# for token in tqdm.tqdm(cars_tokens):
#     index += 1
#     if use_gt:
#         nusc.render_sample(token, out_path = out_dir+str(index)+"_gt.png", verbose=False)
#     else:
#         nusc.render_sample(token, out_path = out_dir+str(index)+"_pred.png", verbose=False)

import os
import json
import argparse
import tqdm
from visual_nuscenes import NuScenes

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize NuScenes results for specific tokens")
    parser.add_argument('--use-gt', action='store_true', help='Use ground truth annotations')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory for visualizations')
    parser.add_argument('--result-json', type=str, help='Path to result JSON file')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to NuScenes dataset')
    parser.add_argument('--token-type', type=str, choices=['cars', 'night', 'rain'], required=True, help='Token group to visualize')

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cars_tokens = ['381fe530586b4c189e10cbefbfb0e773',
 'f0e4797551024f9487a016ee9d9e29e1',
 '72f030bd5dd142d497c090dc04bb5697',
 'dbffcc96729b420380ace0e08ad587d5',
 'cf5c4b437daf4bb3ad68d0f88ea7bff2',
 '561cf9acfda645b79bc52ec82537970c',
 'fd534bf0c87d4d4cb891f0fc6f41a6a9',
 'ff98be19bcf9436cad14f28ff63cb501',
 '4b6fcdc1549847c395d07fefcbdc16eb',
 'b1eea1e3d59e4afdaa6d29481480d974',
 '05cf2a68a2cf4ff1838034f8ecae0c82',
 'b04b95006e24460bacaaf80e299dff01',
 'b006ca82a2654951ae8f42300f66f3c4',
 '3307d1211dfd442db4a6ae544c2ae197',
 '50d86a5cd45440c89e895a5d20ea5267',
 'fe944294f6a44b87a4da95136f86bb79',
 'a9b8176bf0b546a4bc46afc631979805',
 '0e8782aa721545caabc7073d32fb1fb1',
 '8d1c7b2b85fe4befb66efe9e05e0d939',
 'd22de14af8cc4007994efda8354c939b',
 '5f88b9a41f264ebbac7976bf4c796d21',
 '6be3fbcf5c224508adb27b4b703b6625',
 '9e9808fe898648e1b0bbf09b1a30f0d4',
 'ae3d3b5cb4424e7ab501a5d6f0a4980a',
 '88a6b5e35eb44deba6b1065d74b0a3c4',
 'a687809bdfff490198d3b39bfc4bd42b',
 '0a0d6b8c2e884134a3b48df43d54c36a',
 '9e7683e8586542a1b6032980c45f15ce',
 'b9ea04a6121d4a8bb00199b885aa5ef0',
 '08ec4b7fcdb5494c8da174770c1d9245',
 '49823b270a62468397f3265172dcc6bf',
 '4f2578e3107f4ad79f88a61b28893916',
 'f3491dccffc3428aac243872cbfe5072',
 '135bf33890ba4ca2984a931444923eda',
 '5d773ca713f54023b34cd4718a5ee293',
 '20d02a3110fb47348e175a61ad157875',
 '0490bd92372a4a2d98c7136ba6ebcfce',
 '08d95280dd2f42d98a4a6e33dff8e815',
 '4f9ad42bb4a24970b770ba0a87baf47a',
 '33c965ece87842c2ac898543d26dad5f',
 'e32db3f8b6c244c2be60b6d7db30ccc3',
 'f38ef51177c84c2a9fa32584b79e601e',
 'c62b6ac878934607be0c524a02f1692b',
 '6c662b1258e34752be332dcc1eccb7a8',
 '5c7c23122dd94c70b8ab5528f27ba117',
 '7461eb9987354c5db09f9203786b2324',
 '4fc9bbe6198c4b8e837fd2ae0de6a4e2',
 'c48c3dfec8b5454ab97303a00365a77b',
 'c7b18ce7027c4cc3b2068fbf46860b3f',
 '94126983bc0c4de89fb27cefb81f24ef',
 'dafe7793990d41a0ae1e50260899f8b4',
 'c900549f484c4c03872e35d4d927f5d7',
 '868280bdd9bf4ec49d2dd6d4880d4bee',
 'e84d37c300094d51aa2917b4ac006da5',
 'cf2601d5aaee4ab8a700029a33d49be6',
 '4bc57fd309b8495fa222abc3263b47e5',
 '7e034567ff9e4c22819fd59c4cb845da',
 'fc2d1d2139ae40a3b781f70004a8fba8',
 'a7831d4d1db54053a501d0418545fee2',
 '8b57f73177694aee8d393b945df0cd38',
 '2c253adbce79406c962b9382236768da',
 '5b29d1a2b0e646368622dee5e8fa5108',
 '1e82fd0644c142a2be36770ee29815aa',
 '0ad5d20390ec43f59d210c1a3a11b23d',
 'd7bea9a3552b40e29d42324108dff575',
 'a572dd2e95e94e4db66bb5dbfaf870b5',
 'c143943fd6e246448eab88ce1a0aeda9',
 '8c4d447e628c4393839187bcfa0cbd76']

    night_tokens = ['cb4504edb87a40d4b650cd5860f6c3b7',
 'a945d7d02c7a4ef98f1b5aaa1557f546',
 'eb82a835419149aeae82c5e8ea64377b',
 'e74b8ce0b1824e0c8167f2534d8d4f7d',
 '79f6489272c24d3ebc5e225ce6ff2aea',
 'e1b741c0b3a342e789365174a36764f6',
 '1a776100d0974c0b9edc9bb80a842b54',
 'c82417563e44430e9e59224e36872f4e',
 'cce6922361124fd0b4c5f28acf78d65d',
 '1fcc8d01ea7048a9a9eb0eb161778026',
 'fcec74d888d745c0964b785dc309cf2c',
 'f22f8f2fc7314b028419d8332c03a049',
 'ae8a5000d89c4c998a5d6bf95ba2ace9',
 'f161848f032840119785996d26004c77',
 '24dc629e9bf74231affdfc033c79f8d4',
 '5ba9f3b30a5d42aeb790bfca7648bf77',
 '11a46849ff264ac3b75ba0709c4a0a35',
 'bc33987f6a8b47709aa70846e6648877',
 'f9c8bf233bd946ba8425f97ef3ccc5ce',
 '3c8e6743e1ed42139ef8e5fbe3a9be17',
 '79e28a35e07b444c893111137aea81e7',
 'e0a7e1df190e48feb416dad36d140c35',
 'a04be2dd20eb493daa336a299f052426',
 '6a147ca886954584aad7eb68b15db56d',
 'df957c9b8e194e7f88b9a3dc84f12174',
 '3d7733347cc04bfa97a43b4de2e6437f',
 'c3acd43e79464a24b0fadbe1e5ba9950',
 '5622886f3e7e427fbcb0243cf664afe9',
 'de18907d85164692bafbb4916484b9b9',
 '4dad239b9fc048f7a6e69e2ba45b35b1',
 '0657a919d58a462c8905f7a0c706dafb',
 'dd873a094c5b47649b79c3f9dc36dfb2']

    rain_tokens = ['67d2d74087714e4994a68c7347acf55a',
 '98fc274ca14d44a5b9117fbfb04d18b1',
 '668349dd12da4e1fa5d777634af88c32',
 '6fde33ada8b04da9929057fd6b85a72b',
 'bdbc78adca87461d93ce8b90f1f38a1e',
 'dedec80061fd4640be7a8945c9846ff6',
 'a15bd188376049479320d79c39611040',
 '64701483680a4d8eae3c8276b611d8d5',
 '6cefb90da1b1468795b6a3a333c94b38',
 'fdd22a4e963c48fe8987afff20c46648',
 '885d559ea0bb4c75ba6be2ad572f068f',
 '4e9c011236e64ac3876e8ae12290b881',
 'bfe534d72d984d99be6d8f40f7e53085',
 'db29741bf3e2411e953597309225f41c',
 '0cd5782c481c4ef0b4bce413089f04c0',
 '8ba8c0aca3e840cdbd68984c998b1adc',
 'b94fdaf044e2474783bf0d7d18f77f3c',
 '6f91a4053fe841faab7a91a526ab9e66',
 '7604362a89f94113ad7a2bb19a074520',
 'f542a7d55b5c4e97bf42eee1f442356c',
 '1a2db86db8fb409abc7e26578b966d6e',
 'a62913cc73e94f5ca721f41fda89d3ad',
 'a06bcad997144dcb82ca925d698c11cd',
 'c2a4c914bd064ede98dc6b3c193050a8',
 '95f21212afec4736b0abd977b01d5ded',
 'b21b443198bc4552ba7593a471abda1f',
 '73550331af2e48fd8679ad97178ff356',
 '86a69236763d42bf9da6247b8bcd88bb',
 'ed10780136f34011b35e1aaa1d8fb1e5',
 '30507c7761c048c896dc0185e855fd62',
 '3de1d789fd004720bcef0a42d47c1159',
 '7b173aed71f84d58bd218628ea584190',
 '7000c2ccd142491bb54b14e361b9b412',
 '0562be5d8f284bd38d811469f02c857f',
 '2ec05dce9faf45d39cdf8188471e9ea7',
 '098e97621e6b4561a58a74a8bc1a91fd',
 '56a6c2493a4544e3a1f0aaf67db1258d',
 'bb963264d4ef4e6fa80d4f10bb8b2dfa',
 '35afe50d131847739f3221c6ffb88d42',
 '61421b8cc155473c8f1db68c24ebc618']

    if args.use_gt:
        nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=True, pred=False, annotations="sample_annotation")
    else:
        if not args.result_json:
            raise ValueError("result-json is required if not using ground truth")
        nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=True, pred=True, annotations=args.result_json, score_thr=0.25)

    result_path = args.result_json
    with open(f'{result_path}.json') as f:
        table = json.load(f)


    if args.token_type == 'cars':
        selected_tokens = cars_tokens
    elif args.token_type == 'night':
        selected_tokens = night_tokens
    else:
        selected_tokens = rain_tokens


    index=0
    for token in tqdm.tqdm(selected_tokens):
        index += 1
        if args.use_gt:
            nusc.render_sample(token, out_path = args.out_dir+str(index)+"_gt.png", verbose=False)
        else:
            nusc.render_sample(token, out_path = args.out_dir+str(index)+"_pred.png", verbose=False)

if __name__ == '__main__':
    main()
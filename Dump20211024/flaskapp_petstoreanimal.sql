-- MySQL dump 10.13  Distrib 8.0.26, for Win64 (x86_64)
--
-- Host: localhost    Database: flaskapp
-- ------------------------------------------------------
-- Server version	8.0.26

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `petstoreanimal`
--

DROP TABLE IF EXISTS `petstoreanimal`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `petstoreanimal` (
  `IC` varchar(10) NOT NULL,
  `petStoreID` varchar(50) NOT NULL,
  `Name` varchar(50) NOT NULL,
  `PetType` varchar(50) NOT NULL,
  `DateOfBirth` date NOT NULL,
  `Gender` varchar(10) NOT NULL,
  `Vaccindated` varchar(30) NOT NULL,
  `Breed` varchar(50) DEFAULT NULL,
  `Price` double(10,2) NOT NULL,
  `Size` varchar(15) DEFAULT NULL,
  `HDB` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`IC`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `petstoreanimal`
--

LOCK TABLES `petstoreanimal` WRITE;
/*!40000 ALTER TABLE `petstoreanimal` DISABLE KEYS */;
INSERT INTO `petstoreanimal` VALUES ('C0012933W','SG Pet Store','Rainbow lory','Cat','2021-01-14','Male','Vaccinated','British Shorthair',139.15,'',''),('C0098264S','SG Pet Store','Phascogale','Cat','2021-09-26','Male','Vaccinated','Domestic Short Hair',213.11,'',''),('C0472746Q','Paws Shop','Dragon, frilled','Cat','2021-03-01','Female','Not Vaccinated','Scottish Fold',581.25,'',''),('C0735975U','Paws Shop','Emerald green tree boa','Cat','2021-01-15','Female','Not Vaccinated','Exotic Shorthair',842.35,'',''),('C0752216N','Paws Shop','Grizzly bear','Cat','2021-03-28','Female','Not Vaccinated','Domestic Short Hair',888.93,'',''),('C1576061N','Pet Loving Center','Sheep, stone','Cat','2021-07-26','Female','Vaccinated','Ragdoll',311.09,'',''),('C2106079Z','Pet Loving Center','Bohor reedbuck','Cat','2021-08-04','Male','Vaccinated','British Shorthair',315.74,'',''),('C2207913T','Paws Shop','White stork','Cat','2021-04-02','Male','Not Vaccinated','Exotic Shorthair',988.62,'',''),('C2792181Q','Pet Loving Center','Ass, asiatic wild','Cat','2021-09-15','Female','Not Vaccinated','Munchkin',212.46,'',''),('C3160614E','Pet Loving Center','Plover, three-banded','Cat','2021-02-07','Female','Not Vaccinated','Ragdoll',212.80,'',''),('C3286923E','Pet Loving Center','Coatimundi, ring-tailed','Cat','2021-03-07','Male','Not Vaccinated','Ragdoll',413.53,'',''),('C5118959R','Pet Loving Center','Wambenger, red-tailed','Cat','2021-03-22','Female','Vaccinated','Maine Coon',638.21,'',''),('C5845881I','SG Pet Store','Burchell gonolek','Cat','2021-06-29','Female','Vaccinated','Norwegian Forest Cat',123.56,'',''),('C6419414M','Pet Loving Center','Baboon, olive','Cat','2021-02-17','Female','Not Vaccinated','Ragdoll',509.32,'',''),('C6449170V','Pet Loving Center','Sloth bear','Cat','2021-03-10','Male','Not Vaccinated','Maine Coon',629.14,'',''),('C6714346C','SG Pet Store','Caracara, yellow-headed','Cat','2021-04-12','Male','Not Vaccinated','Munchkin',226.72,'',''),('C6742514B','Pet Loving Center','Asian elephant','Cat','2021-03-06','Female','Not Vaccinated','Ragdoll',396.27,'',''),('C7930422M','Paws Shop','Monster, gila','Cat','2021-10-13','Male','Vaccinated','Norwegian Forest Cat',288.36,'',''),('C8107123B','Paws Shop','Monkey, vervet','Cat','2021-06-30','Female','Vaccinated','Persian',354.60,'',''),('C8324760S','SG Pet Store','Downy woodpecker','Cat','2021-07-24','Male','Vaccinated','Norwegian Forest Cat',578.73,'',''),('C8531155V','Paws Shop','Little blue penguin','Cat','2021-06-09','Female','Not Vaccinated','Persian',858.57,'',''),('C8668641W','SG Pet Store','River wallaby','Cat','2021-09-11','Female','Vaccinated','Maine Coon',974.90,'',''),('C8718697Q','Pet Loving Center','Racer, american','Cat','2021-05-15','Female','Vaccinated','British Shorthair',324.37,'',''),('C8814176F','Pet Loving Center','Boa, cook tree','Cat','2021-10-06','Female','Not Vaccinated','Exotic Shorthair',547.38,'',''),('C8990491N','SG Pet Store','Porcupine, crested','Cat','2021-09-26','Female','Vaccinated','Exotic Shorthair',438.18,'',''),('C9094417O','Paws Shop','Cardinal, red-capped','Cat','2021-10-01','Male','Not Vaccinated','Munchkin',765.97,'',''),('C9400122Q','Pet Loving Center','Skua, great','Cat','2021-02-10','Male','Not Vaccinated','Domestic Short Hair',573.28,'',''),('C9488518Y','SG Pet Store','Common green iguana','Cat','2021-08-03','Male','Vaccinated','Munchkin',619.91,'',''),('C9575969Q','Paws Shop','Porcupine, tree','Cat','2021-05-21','Male','Not Vaccinated','British Shorthair',206.78,'',''),('C9700425J','Pet Loving Center','Eagle, pallas fish','Cat','2021-03-16','Male','Not Vaccinated','Domestic Short Hair',941.24,'',''),('C9733889Q','SG Pet Store','Grenadier','Cat','2021-07-11','Female','Vaccinated','Munchkin',628.83,'',''),('C9754668K','SG Pet Store','Bird, bare-faced go away','Cat','2021-05-17','Female','Not Vaccinated','Scottish Fold',376.92,'',''),('C9848326R','SG Pet Store','Vulture, white-headed','Cat','2021-08-22','Male','Vaccinated','Exotic Shorthair',308.32,'',''),('C9965860X','Pet Loving Center','Phascogale, brush-tailed','Cat','2021-06-06','Male','Vaccinated','Ragdoll',335.48,'',''),('D0029449T','Paws Shop','Penguin, fairy','Dog','2021-02-13','Female','Vaccinated','Yorkshire Terrier',2272.14,'Small','HDB'),('D0030016J','Pet Loving Center','Bat, little brown','Dog','2021-07-01','Female','Not Vaccinated','West Highland Terrier',2837.78,'Small','HDB'),('D0035868W','Paws Shop','Bleeding heart monkey','Dog','2021-01-31','Male','Not Vaccinated','Boston Terrier',2971.84,'Small','HDB'),('D0052420M','Paws Shop','Jackal, indian','Dog','2021-05-23','Male','Vaccinated','Japanese Chin',933.42,'Small','HDB'),('D0072844W','Paws Shop','Dragon, komodo','Dog','2021-01-26','Male','Not Vaccinated','Italian Greyhound',2112.70,'Small','HDB'),('D0117912T','Pet Loving Center','Lark, horned','Dog','2021-06-09','Female','Vaccinated','Border Collie',1246.65,'Large',''),('D0119808I','SG Pet Store','Southern sea lion','Dog','2021-01-21','Male','Not Vaccinated','Cardigan Welsh Corgi',2748.97,'Small',''),('D0136036I','SG Pet Store','Lesser mouse lemur','Dog','2021-04-26','Female','Vaccinated','Boston Terrier',777.74,'Small','HDB'),('D0156638E','Paws Shop','Red-billed buffalo weaver','Dog','2021-07-03','Male','Not Vaccinated','Lakeland Terrier',2845.12,'Small','HDB'),('D0160378J','SG Pet Store','North American river otter','Dog','2021-01-13','Female','Not Vaccinated','Lakeland Terrier',2302.11,'Small','HDB'),('D0198019S','Pet Loving Center','Burrowing owl','Dog','2021-08-12','Male','Not Vaccinated','Lhasa Apso',2924.87,'Small','HDB'),('D0205877Z','Pet Loving Center','Galapagos penguin','Dog','2021-03-06','Female','Vaccinated','Jack Russell Terrier',2569.58,'Small','HDB'),('D0285317B','SG Pet Store','Woodrat (unidentified)','Dog','2021-03-09','Male','Vaccinated','British Bulldog',2205.64,'Small',''),('D0322054P','SG Pet Store','Stork, marabou','Dog','2021-02-18','Female','Vaccinated','Golden Retriever',1579.18,'Large',''),('D0330302X','Paws Shop','Hornbill, yellow-billed','Dog','2021-08-06','Female','Vaccinated','Smooth Fox Terrier',2327.33,'Small','HDB'),('D0365664N','Pet Loving Center','Arctic tern','Dog','2021-02-17','Female','Vaccinated','Bichon Frise',2403.12,'Small','HDB'),('D0398979K','Pet Loving Center','Prehensile-tailed porcupine','Dog','2021-03-05','Male','Not Vaccinated','Silky Terrier',2085.91,'Small','HDB'),('D0436249B','Paws Shop','Otter, brazilian','Dog','2021-07-19','Female','Not Vaccinated','Lhasa Apso',2667.36,'Small','HDB'),('D0457940K','Paws Shop','Turaco, violet-crested','Dog','2021-05-02','Male','Not Vaccinated','Yorkshire Terrier',877.07,'Small','HDB'),('D0518880K','Pet Loving Center','Fairy penguin','Dog','2021-07-16','Male','Not Vaccinated','Pekingese',2790.21,'Small','HDB'),('D0534803V','SG Pet Store','Bahama pintail','Dog','2021-08-14','Male','Vaccinated','Toy Poodle',2332.38,'Small','HDB'),('D0542953M','Paws Shop','Silver gull','Dog','2021-03-07','Male','Vaccinated','Shiba Inu',1902.25,'Large',''),('D0597579J','Pet Loving Center','Alligator, mississippi','Dog','2021-08-19','Male','Not Vaccinated','Lhasa Apso',1664.69,'Small','HDB'),('D0676721O','Paws Shop','Openbill, asian','Dog','2021-07-28','Male','Not Vaccinated','Sealyham Terrier',2302.59,'Small','HDB'),('D0754225S','SG Pet Store','Caracal','Dog','2021-10-05','Female','Vaccinated','Pug',1626.65,'Small','HDB'),('D0791233J','SG Pet Store','California sea lion','Dog','2021-08-20','Male','Not Vaccinated','Border Collie',1750.17,'Large',''),('D0800736Q','SG Pet Store','Gonolek, burchell','Dog','2021-10-12','Female','Vaccinated','Cardigan Welsh Corgi',856.52,'Small',''),('D0865779F','SG Pet Store','White-nosed coatimundi','Dog','2021-07-04','Male','Not Vaccinated','Tibetan Mastiff',930.07,'Large',''),('D0937584U','Pet Loving Center','Weaver, sociable','Dog','2021-04-17','Male','Not Vaccinated','Maltese',2913.33,'Small','HDB'),('D0956219O','Paws Shop','Possum, ring-tailed','Dog','2021-04-05','Male','Vaccinated','Poodle',1869.90,'Large',''),('D1030068P','Pet Loving Center','Green-winged macaw','Dog','2021-09-09','Female','Not Vaccinated','Japanese Spitz',1710.47,'Small','HDB'),('D1119186X','Pet Loving Center','House crow','Dog','2021-06-17','Female','Not Vaccinated','Smooth Fox Terrier',1061.57,'Small','HDB'),('D1566900C','SG Pet Store','Rhea, common','Dog','2021-09-20','Female','Not Vaccinated','Papillon',754.70,'Small','HDB'),('D1650231P','Pet Loving Center','Cockatoo, red-breasted','Dog','2021-08-22','Female','Not Vaccinated','Tibetan Mastiff',1397.28,'Large',''),('D1709745B','Paws Shop','Lion, african','Dog','2021-10-08','Female','Not Vaccinated','Shih Tzu',2459.75,'Small','HDB'),('D1841887S','SG Pet Store','Galapagos sea lion','Dog','2021-06-22','Female','Vaccinated','Shih Tzu',779.16,'Small','HDB'),('D1868165O','Pet Loving Center','Gull, pacific','Dog','2021-06-13','Female','Vaccinated','Chow Chow',500.54,'Large',''),('D1893837Z','Paws Shop','Long-tailed jaeger','Dog','2021-07-10','Male','Vaccinated','Golden Retriever',1557.07,'Large',''),('D1925186B','Paws Shop','Pallas fish eagle','Dog','2021-03-20','Female','Vaccinated','Siberian Husky',2344.59,'Large',''),('D1933942V','Paws Shop','Hornbill, red-billed','Dog','2021-04-15','Female','Not Vaccinated','Bichon Frise',613.59,'Small','HDB'),('D2128733X','SG Pet Store','Violet-crested turaco','Dog','2021-06-07','Female','Vaccinated','French Bulldog',606.87,'Small',''),('D2134457Q','SG Pet Store','Native cat','Dog','2021-01-15','Male','Vaccinated','Shetland Sheepdog',539.95,'Large','HDB'),('D2286230S','Paws Shop','Lilac-breasted roller','Dog','2021-06-09','Male','Vaccinated','Border Terrier',1906.24,'Small','HDB'),('D2300806M','Pet Loving Center','Trotter, lily','Dog','2021-08-29','Female','Vaccinated','Cavalier King Charles Spaniel',1964.08,'Small','HDB'),('D2414883W','SG Pet Store','Hartebeest, red','Dog','2021-04-19','Female','Vaccinated','Griffon Brabancon',940.91,'Small','HDB'),('D2481196J','SG Pet Store','Deer, black-tailed','Dog','2021-08-05','Female','Vaccinated','Border Terrier',1723.90,'Small','HDB'),('D2681266I','SG Pet Store','Steenbok','Dog','2021-07-04','Female','Vaccinated','Pembroke Welsh Corgi',2747.91,'Small',''),('D2681332A','SG Pet Store','Huron','Dog','2021-10-18','Female','Not Vaccinated','Chihuahua',2941.81,'Small','HDB'),('D2806437P','Paws Shop','Long-finned pilot whale','Dog','2021-05-19','Female','Not Vaccinated','Affenpinscher',2020.27,'Small','HDB'),('D2861320A','Pet Loving Center','Yellow-brown sungazer','Dog','2021-03-16','Female','Vaccinated','Pomeranian',2999.56,'Small','HDB'),('D2909112Q','Pet Loving Center','Meerkat, red','Dog','2021-08-23','Male','Vaccinated','Shih Tzu',1389.93,'Small','HDB'),('D2949473O','SG Pet Store','Anaconda (unidentified)','Dog','2021-09-11','Male','Not Vaccinated','Silky Terrier',645.28,'Small','HDB'),('D2978674A','Pet Loving Center','African pied wagtail','Dog','2021-05-19','Female','Not Vaccinated','Boston Terrier',1415.81,'Small','HDB'),('D3060849G','Paws Shop','Civet, small-toothed palm','Dog','2021-03-23','Male','Not Vaccinated','English Toy Spaniel',898.48,'Small','HDB'),('D3237684H','Pet Loving Center','Tiger','Dog','2021-10-04','Female','Vaccinated','Schipperkee',893.19,'Small','HDB'),('D3239085V','Paws Shop','Tarantula, salmon pink bird eater','Dog','2021-09-04','Male','Vaccinated','Shiba Inu',689.95,'Large',''),('D3286394S','SG Pet Store','Dragon, asian water','Dog','2021-03-15','Male','Not Vaccinated','Chow Chow',1855.99,'Large',''),('D3469552X','Pet Loving Center','Francolin, swainson','Dog','2021-07-19','Male','Not Vaccinated','Labrador Retriever',518.55,'Large',''),('D3513649O','Pet Loving Center','Netted rock dragon','Dog','2021-03-13','Male','Not Vaccinated','Shiba Inu',631.05,'Large',''),('D3534171F','Paws Shop','Egyptian cobra','Dog','2021-09-13','Male','Not Vaccinated','Brussels Griffon',1457.23,'Small','HDB'),('D3588079P','Pet Loving Center','Barking gecko','Dog','2021-09-29','Female','Not Vaccinated','Japanese Spitz',1827.40,'Small','HDB'),('D3648762E','Pet Loving Center','Gecko','Dog','2021-05-12','Female','Not Vaccinated','Schipperkee',2947.40,'Small','HDB'),('D3790882Y','SG Pet Store','Magnificent frigate bird','Dog','2021-05-22','Female','Not Vaccinated','Border Collie',603.58,'Large',''),('D4012025T','Pet Loving Center','Quail, gambel','Dog','2021-08-22','Male','Vaccinated','Siberian Husky',2963.58,'Large',''),('D4036528C','Pet Loving Center','Stork, greater adjutant','Dog','2021-08-25','Male','Not Vaccinated','Japanese Chin',2976.16,'Small','HDB'),('D4364308V','Pet Loving Center','Squirrel, golden-mantled ground','Dog','2021-04-06','Male','Not Vaccinated','Manchester Terrier',2316.97,'Large','HDB'),('D4477360I','Pet Loving Center','Gull, silver','Dog','2021-10-11','Male','Not Vaccinated','Australian Terrier',1122.69,'Small','HDB'),('D4609332Q','Paws Shop','Osprey','Dog','2021-02-01','Male','Not Vaccinated','Miniature Pinscher',540.35,'Small','HDB'),('D5006563F','SG Pet Store','Eastern cottontail','Dog','2021-07-11','Male','Not Vaccinated','Pug',521.27,'Small','HDB'),('D5033362G','SG Pet Store','Brown brocket','Dog','2021-10-12','Male','Vaccinated','Cocker Spaniel',1444.85,'Large',''),('D5064376V','Pet Loving Center','Capuchin, brown','Dog','2021-09-22','Female','Vaccinated','Boston Terrier',1736.58,'Small','HDB'),('D5131246E','Paws Shop','Cat, jungle','Dog','2021-05-26','Female','Not Vaccinated','Silky Terrier',2247.30,'Small','HDB'),('D5139404M','SG Pet Store','Giant anteater','Dog','2021-02-13','Male','Vaccinated','Shetland Sheepdog',2024.26,'Large','HDB'),('D5142716D','Pet Loving Center','Puma, south american','Dog','2021-10-18','Female','Vaccinated','Scottish Terrier',523.72,'Small','HDB'),('D5271590Y','Pet Loving Center','Grenadier, common','Dog','2021-01-31','Male','Vaccinated','Brussels Griffon',1479.48,'Small','HDB'),('D5430197O','SG Pet Store','Red and blue macaw','Dog','2021-06-25','Female','Vaccinated','Dachshund',2262.91,'Small','HDB'),('D5444994F','Pet Loving Center','Bird, magnificent frigate','Dog','2021-02-01','Female','Vaccinated','Pembroke Welsh Corgi',2741.79,'Small',''),('D5459240O','Paws Shop','Hyena, striped','Dog','2021-05-23','Male','Not Vaccinated','Border Collie',2028.63,'Large',''),('D5490742A','Pet Loving Center','Civet cat','Dog','2021-02-25','Female','Not Vaccinated','Cocker Spaniel',1143.13,'Large',''),('D5652221I','Paws Shop','Snowy owl','Dog','2021-01-20','Male','Not Vaccinated','Miniature Pinscher',1362.87,'Small','HDB'),('D5889271F','Paws Shop','Lily trotter','Dog','2021-03-31','Male','Not Vaccinated','Cardigan Welsh Corgi',2393.27,'Small',''),('D6121947M','Paws Shop','Chimpanzee','Dog','2021-02-21','Male','Vaccinated','Border Terrier',1942.57,'Small','HDB'),('D6158453O','Pet Loving Center','Yellow-rumped siskin','Dog','2021-03-25','Male','Not Vaccinated','English Toy Spaniel',724.14,'Small','HDB'),('D6216780V','SG Pet Store','Ibis, sacred','Dog','2021-08-10','Male','Not Vaccinated','Maltese',592.43,'Small','HDB'),('D6289755U','Paws Shop','Macaque, pig-tailed','Dog','2021-09-29','Male','Vaccinated','Cairn Terrier',1959.66,'Small','HDB'),('D6341262Z','Pet Loving Center','Coot, red-knobbed','Dog','2021-07-10','Female','Not Vaccinated','Bichon Frise',2253.05,'Small','HDB'),('D6409621V','SG Pet Store','Wolf, mexican','Dog','2021-06-23','Male','Not Vaccinated','Pug',2370.16,'Small','HDB'),('D6412979J','Pet Loving Center','Roan antelope','Dog','2021-05-19','Female','Not Vaccinated','English Toy Spaniel',1602.47,'Small','HDB'),('D6425136X','Pet Loving Center','Turtle (unidentified)','Dog','2021-01-07','Male','Vaccinated','English Toy Spaniel',2483.41,'Small','HDB'),('D6554415G','Pet Loving Center','Lizard, desert spiny','Dog','2021-06-24','Male','Vaccinated','Boston Terrier',532.08,'Small','HDB'),('D6607919K','Pet Loving Center','Whale, long-finned pilot','Dog','2021-04-22','Female','Vaccinated','Pug',2679.37,'Small','HDB'),('D6651460M','Pet Loving Center','Tortoise, galapagos','Dog','2021-09-03','Female','Not Vaccinated','Silky Terrier',698.81,'Small','HDB'),('D6655406T','SG Pet Store','Cape fox','Dog','2021-07-20','Female','Not Vaccinated','Norfolk Terrier',579.41,'Small','HDB'),('D6688626Y','Paws Shop','Zorilla','Dog','2021-09-06','Female','Vaccinated','Border Collie',2688.00,'Large',''),('D6777908M','Paws Shop','Pine squirrel','Dog','2021-06-23','Male','Vaccinated','Cocker Spaniel',2465.35,'Large',''),('D6787512W','Pet Loving Center','Shelduck, european','Dog','2021-07-22','Male','Not Vaccinated','Scottish Terrier',1082.43,'Small','HDB'),('D7044895P','Pet Loving Center','Colobus, black and white','Dog','2021-08-04','Female','Vaccinated','Schipperkee',1802.15,'Small','HDB'),('D7168862K','Paws Shop','South American meadowlark (unidentified)','Dog','2021-02-03','Male','Vaccinated','Affenpinscher',1152.11,'Small','HDB'),('D7222131H','SG Pet Store','Eastern boa constrictor','Dog','2021-02-02','Female','Vaccinated','English Toy Spaniel',1749.76,'Small','HDB'),('D7251867K','Paws Shop','Eastern cottontail rabbit','Dog','2021-01-18','Female','Vaccinated','Scottish Terrier',1939.61,'Small','HDB'),('D7254400C','Pet Loving Center','Skink, african','Dog','2021-03-14','Male','Vaccinated','Chihuahua',1143.13,'Small','HDB'),('D7378649M','Paws Shop','Common eland','Dog','2021-06-10','Female','Not Vaccinated','Miniature Schnauzer',2390.94,'Small','HDB'),('D7393853M','Pet Loving Center','Dove, emerald-spotted wood','Dog','2021-03-09','Female','Not Vaccinated','Chinese Crested',743.44,'Small','HDB'),('D7467433T','Paws Shop','Woodpecker, downy','Dog','2021-03-26','Male','Not Vaccinated','Yorkshire Terrier',2542.80,'Small','HDB'),('D7467700A','SG Pet Store','White-throated toucan','Dog','2021-04-26','Female','Not Vaccinated','Pembroke Welsh Corgi',1029.56,'Small',''),('D7481094C','Paws Shop','Red-capped cardinal','Dog','2021-04-17','Female','Vaccinated','Affenpinscher',2542.25,'Small','HDB'),('D7641284H','Paws Shop','Malabar squirrel','Dog','2021-03-05','Male','Not Vaccinated','Siberian Husky',1829.33,'Large',''),('D8036589F','Paws Shop','Whale, baleen','Dog','2021-06-01','Male','Vaccinated','Italian Greyhound',2454.84,'Small','HDB'),('D8289395U','Paws Shop','Chital','Dog','2021-03-05','Female','Not Vaccinated','Jack Russell Terrier',1351.54,'Small','HDB'),('D8328818A','Pet Loving Center','Langur, common','Dog','2021-08-10','Male','Not Vaccinated','Labrador Retriever',1779.53,'Large',''),('D8392059Q','SG Pet Store','Ring-tailed coatimundi','Dog','2021-01-14','Female','Vaccinated','Brussels Griffon',1122.76,'Small','HDB'),('D8489887L','SG Pet Store','Sportive lemur','Dog','2021-02-23','Female','Vaccinated','Cavalier King Charles Spaniel',2905.01,'Small','HDB'),('D8506607B','Pet Loving Center','Squirrel, european red','Dog','2021-10-08','Male','Vaccinated','Chihuahua',1714.21,'Small','HDB'),('D8529284A','Paws Shop','Long-crested hawk eagle','Dog','2021-05-13','Male','Vaccinated','Labrador Retriever',2104.89,'Large',''),('D8783107H','SG Pet Store','Eagle, tawny','Dog','2021-05-29','Male','Vaccinated','Italian Greyhound',924.73,'Small','HDB'),('D8814718U','SG Pet Store','Honey badger','Dog','2021-03-25','Male','Not Vaccinated','Dandie Dinmont Terrier',2879.46,'Small','HDB'),('D8997222I','Paws Shop','Stone sheep','Dog','2021-05-20','Male','Not Vaccinated','Pug',2849.48,'Small','HDB'),('D9077571E','Pet Loving Center','Fox, bat-eared','Dog','2021-08-12','Male','Not Vaccinated','Mexican Hairless Dog',958.10,'Large',''),('D9082325H','Paws Shop','Macaw, red and blue','Dog','2021-07-08','Female','Not Vaccinated','British Bulldog',1125.50,'Small',''),('D9106566A','Paws Shop','Numbat','Dog','2021-02-13','Male','Vaccinated','Pembroke Welsh Corgi',1221.55,'Small',''),('D9109809A','Paws Shop','Bleu, blue-breasted cordon','Dog','2021-06-14','Female','Vaccinated','Chinese Crested',1176.03,'Small','HDB'),('D9133175K','Pet Loving Center','Savanna fox','Dog','2021-08-03','Female','Vaccinated','Silky Terrier',2678.74,'Small','HDB'),('D9338167C','Paws Shop','Guanaco','Dog','2021-02-23','Female','Not Vaccinated','Cavalier King Charles Spaniel',1510.19,'Small','HDB'),('D9351237E','SG Pet Store','Red-knobbed coot','Dog','2021-04-05','Male','Not Vaccinated','Jack Russell Terrier',1667.81,'Small','HDB'),('D9577557Y','Pet Loving Center','Snake, buttermilk','Dog','2021-08-23','Male','Not Vaccinated','Boston Terrier',880.63,'Small','HDB'),('D9597572S','Paws Shop','Quoll, eastern','Dog','2021-05-24','Male','Not Vaccinated','Brussels Griffon',2608.82,'Small','HDB'),('D9615113Z','SG Pet Store','Reedbuck, bohor','Dog','2021-06-28','Male','Vaccinated','Siberian Husky',1510.16,'Large',''),('D9776852M','SG Pet Store','Iguana','Dog','2021-09-21','Male','Vaccinated','Border Terrier',1796.50,'Small','HDB'),('D9795783C','Paws Shop','Hawk-eagle, crowned','Dog','2021-09-13','Female','Vaccinated','Dachshund',2547.55,'Small','HDB'),('D9866784W','Paws Shop','Wild boar','Dog','2021-08-10','Female','Vaccinated','Maltese',2483.55,'Small','HDB'),('D9899085E','SG Pet Store','Burrowing','Dog','2021-09-27','Female','Vaccinated','West Highland Terrier',2285.39,'Small','HDB'),('D9955090Y','Paws Shop','Great white','Dog','2021-06-16','Male','Vaccinated','Tibetan Mastiff',1021.91,'Large',''),('D9958495R','Pet Loving Center','Butterfly, canadian tiger swallowtail','Dog','2021-09-04','Female','Not Vaccinated','British Bulldog',619.72,'Small','');
/*!40000 ALTER TABLE `petstoreanimal` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2021-10-24 19:56:27

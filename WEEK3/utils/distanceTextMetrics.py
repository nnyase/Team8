import textdistance

def getDistance2Strings(str1,str2,formula):
    if formula== 1:   #"hamming":
        result=textdistance.hamming.normalized_distance(str1, str2)
    elif formula==2:#"mlipns":
        result=textdistance.mlipns.normalized_distance(str1, str2)
    elif formula==3:#"levenshtein":
        result=textdistance.levenshtein.normalized_distance(str1, str2)
    elif formula==4:#"damerau-levenshtein":
        result=textdistance.damerau_levenshtein.normalized_distance(str1, str2)
    elif formula==5:#"jaro_winkler":
        result=textdistance.jaro_winkler.normalized_distance(str1, str2)
    elif formula==6:#"strcmp95":
        result=textdistance.strcmp95.normalized_distance(str1, str2)
    elif formula==7:#"needleman-wunsch":
        result=textdistance.needleman_wunsch.normalized_distance(str1, str2)     
    elif formula==8:#"gotoh":
        result=textdistance.gotoh.normalized_distance(str1, str2)
    elif formula==9:#"smith-Waterman":
        result=textdistance.smith_waterman.normalized_distance(str1, str2)           
    elif formula==10:#"jaccard_index":
        result=textdistance.jaccard.normalized_distance(str1, str2)                     
    elif formula==11:#"sorensen_Dice_coefficient":
        result=textdistance.sorensen_dice.normalized_distance(str1, str2)
    elif formula==12:#"tversky_index":
        result=textdistance.tversky.normalized_distance(str1, str2)
    elif formula==13:#"overlap_coefficient":
        result=textdistance.overlap.normalized_distance(str1, str2)     
    elif formula==14:#"tanimoto_distance":
        result=textdistance.tanimoto.normalized_distance(str1, str2)
    elif formula==15:#"cosine_similarity":
        result=textdistance.cosine.normalized_distance(str1, str2)           
    elif formula==16:#"monge_elkan":
        result=textdistance.monge_elkan.normalized_distance(str1, str2)                     
    elif formula==17:#"bag_distance":
        result=textdistance.bag.normalized_distance(str1, str2)  
    elif formula==18:#"longest_common_subsequence_similarity":
        result=textdistance.lcsseq.normalized_distance(str1, str2)           
    elif formula==19:#"longest_common_substring_similarity":
        result=textdistance.lcsstr.normalized_distance(str1, str2)                     
    elif formula==20:#"Ratcliff-Obershelp similarity":
        result=textdistance.ratcliff_obershelp.normalized_distance(str1, str2)
    elif formula==21:#"Arithmetic coding":
        result=textdistance.arith_ncd.normalized_distance(str1, str2)
    elif formula==22:#"RLE":
        result=textdistance.rle_ncd.normalized_distance(str1, str2)     
    elif formula==23:#"BWT RLE":
        result=textdistance.bwtrle_ncd.normalized_distance(str1, str2)
    elif formula==24:#"Square Root":
        result=textdistance.sqrt_ncd.normalized_distance(str1, str2)           
    elif formula==25:#"Entropy":
        result=textdistance.entropy_ncd.normalized_distance(str1, str2)                     
    elif formula==26:#"BZ2":
        result=textdistance.bz2_ncd.normalized_distance(str1, str2)   
    elif formula==27:#"LZMA":
        result=textdistance.lzma_ncd.normalized_distance(str1, str2)           
    elif formula==28:#"ZLib":
        result=textdistance.zlib_ncd.normalized_distance(str1, str2)                     
    elif formula==29:#"MRA":
        result=textdistance.mra.normalized_distance(str1, str2)
    elif formula==30:#"Editex":
        result=textdistance.editex.normalized_distance(str1, str2)
    elif formula==31:#/"Prefix similarity":
        result=textdistance.prefix.normalized_distance(str1, str2)     
    elif formula==32:#"Postfix similarity":
        result=textdistance.postfix.normalized_distance(str1, str2)
    elif formula==33:#"Length distance":
        result=textdistance.length.normalized_distance(str1, str2)           
    elif formula==34:#"Identity similarity":
        result=textdistance.identity.normalized_distance(str1, str2)                     
    elif formula==35:#"Matrix similarity":
        result=textdistance.matrix.normalized_distance(str1, str2) 
                         
    return result

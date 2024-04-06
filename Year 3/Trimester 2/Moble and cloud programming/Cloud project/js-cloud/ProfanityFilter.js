function AppendText(word, shouldCensor)
{
    if (shouldCensor)
        return '*'.repeat(word.length) + " ";
    return word + " ";
}

// Edit this function to fit with the server fit the requirement of the server
function FilterText()
{
    const msg = document.getElementById('userInput').value;
    var words = msg.toLowerCase().split(/\s+/);
    var filteredText = FilterProfanity(words);
    filteredText = MaskNRIC(filteredText.split(/\s+/));
    document.getElementById('output').textContent = `${filteredText}`;
}

function MaskNRIC(words)
{
    /* 
        ^: Asserts the start of the string.
        [STFG]: Matches any one character from the set STFG, which are the possible starting letters of an NRIC.
        \d{7}: Matches exactly 7 digits.
        [A-Z]: Matches any uppercase alphabet.
        $: Asserts the end of the string.
     */
    var filteredText = "";
    const nricRegex = /^[stfg]\d{7}[a-z]$/;
    for (const w of words)
        filteredText += AppendText(w, nricRegex.test(w));
    return filteredText;
}

function FilterProfanity(words)
{
    const reEscape = s => s.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');       
    const badWords = [
        'fuck', 'bitch', 'fk', 'bij', 'cb', 'cheebai', 'jibai', 'knn', 'useless', 'shit', 'lanjiao', 'sohai', 
        'pukimak', 'anjing', 'babi', 'motherfucker', 'motherfker', 'noob', 'n00b', 'stupid', 'stoopid', 'st00pid',
        'dumb', 'asshole', 'ass', '@ss', 'b!tch', 'b!j', '@sshole', '@ssh0le', 'lj', 'dumbfk', 'lampa', 'bullshit',
        'bullsh!t', 'bloody', 'hell', 'bodoh', 'bod0h', 'b0doh', 'b0d0h', 'penis', 'pussy', 'vagina', 'cock', 'chicken',
        'no balls', 'nigger', 'nigg', 'nigga', 'nigg@', 'stfu', 'shut up', 'faggot', 'f@ggot', 'f@gg0t', 'fagg0t', 'cunt',
    ];
    const badWordsRE = new RegExp(badWords.map(reEscape).join('|'));

    var filteredText = "";
    for (const w of words)
        filteredText += AppendText(w, w.match(badWordsRE));
    return filteredText;
}
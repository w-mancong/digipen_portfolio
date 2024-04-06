const fs = require('node:fs/promises');
const ProfanityFilter = require('./ProfanityFilter')

async function example(msg)
{
    try {
        const profanityFilter = new ProfanityFilter();
        const data = await fs.readFile('profanity_words.txt', { encoding: 'utf8' });
        for (const w of data)
            profanityFilter.addWord(w);
    } catch (err) {
        console.log(err);
    }
}

example("I fuck you");